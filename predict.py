import json
import os
import random
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from cog import BaseModel, BasePredictor, Input, Path
import requests
import websocket

COMFY_HOST = "127.0.0.1:8188"
COMFY_PYTHON = "/root/comfy_env/bin/python"
CHECKPOINT_URL = "https://huggingface.co/Comfy-Org/YuE2/resolve/main/checkpoints/yue2_3b_int8_convrot.safetensors"
CHECKPOINT_PATH = (
    "/root/ComfyUI/models/checkpoints/yue2_3b_int8_convrot.safetensors"
)

DEFAULT_STYLE = (
    "late-night smooth jazz radio station bumper, smoky tenor saxophone, "
    "warm rhodes electric piano chords, brush snare, 75 bpm, deep resonant male"
    " vocals"
)

DEFAULT_LYRICS = """[verse]
When city shadows turn to blue,
We play the midnight sound for you.
[chorus]
Slip into velvet, ease your mind,
The smoothest rhythm you can find."""


class Output(BaseModel):
  audio: Path
  score_abc: str


class Predictor(BasePredictor):

  def setup(self):
    """Downloads weights if missing and boots ComfyUI in its dedicated virtualenv."""
    # 1. Download model weights on boot over Replicate's high-speed datacenter pipe
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    if (
        not os.path.exists(CHECKPOINT_PATH)
        or os.path.getsize(CHECKPOINT_PATH) < 3_500_000_000
    ):
      print(f"Downloading YuE2 INT8 checkpoint (~3.96 GB) to {CHECKPOINT_PATH}...")
      # --progress=dot:giga outputs 1 dot per GB to prevent log buffer clipping
      subprocess.run(
          ["wget", "-c", "--progress=dot:giga", CHECKPOINT_URL, "-O", CHECKPOINT_PATH],
          check=True,
      )
      print("Checkpoint download complete.")
    else:
      print("Checkpoint already present in container cache.")

    # 2. Start ComfyUI headless server in the background
    print("Starting background ComfyUI instance in isolated virtualenv...")
    cmd = [
        COMFY_PYTHON,
        "/root/ComfyUI/main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        "8188",
        "--fast",
        "fp16_accumulation",
        "--verbose",
        "INFO",
    ]
    self.comfy_process = subprocess.Popen(cmd)

    # 3. Poll until ComfyUI responds on port 8188
    ready = False
    for _ in range(60):
      if self.comfy_process.poll() is not None:
        raise RuntimeError(
            f"ComfyUI process exited prematurely with code"
            f" {self.comfy_process.returncode}"
        )

      try:
        res = requests.get(f"http://{COMFY_HOST}/system_stats", timeout=1)
        if res.status_code == 200:
          ready = True
          break
      except Exception:
        time.sleep(1)

    if not ready:
      raise RuntimeError("ComfyUI failed to start within 60 seconds.")
    print("ComfyUI server is online and ready for jobs.")

  def predict(
      self,
      style: str = Input(
          description="Genre, instruments, mood, tempo, vocal character",
          default=DEFAULT_STYLE,
      ),
      lyrics: str = Input(
          description=(
              "Song lyrics or bracketed musical structure tags ([verse],"
              " [chorus], etc.)"
          ),
          default=DEFAULT_LYRICS,
      ),
      audio_format: str = Input(
          description="Audio output format",
          choices=["mp3", "wav", "flac"],
          default="mp3",
      ),
      cot: str = Input(
          description=(
              "Planning mode: 'full' (melody + chords), 'melody' (melody only),"
              " 'off' (direct synthesis)"
          ),
          choices=["full", "melody", "off"],
          default="full",
      ),
      max_duration: float = Input(
          description="Target song duration in seconds",
          ge=15.0,
          le=360.0,
          default=30.0,
      ),
      temperature: float = Input(
          description="Creativity and randomness of generation",
          ge=0.1,
          le=2.0,
          default=1.0,
      ),
      steps: int = Input(
          description=(
              "Acoustic diffusion steps (15-20 fast, 25-32 standard, 50 max)"
          ),
          ge=10,
          le=100,
          default=32,
      ),
      sampler_name: str = Input(
          description="Diffusion ODE solver algorithm",
          choices=["dpm_2", "euler", "dpmpp_2m"],
          default="dpm_2",
      ),
      scheduler: str = Input(
          description="Noise reduction schedule curve",
          choices=["sgm_uniform", "karras", "simple"],
          default="sgm_uniform",
      ),
      max_abc_tokens: int = Input(
          description="Max tokens for the ABC musical score planning stage",
          ge=256,
          le=8192,
          default=8192,
      ),
      custom_abc: str = Input(
          description=(
              "(Optional) Pre-written ABC score. Bypasses the symbolic planner"
              " if provided."
          ),
          default=None,
      ),
      seed: int = Input(
          description="Random seed for reproducibility (-1 for random)",
          default=-1,
      ),
  ) -> Output:
    """Modifies the dual-stage node graph and executes ComfyUI headless."""
    # 1. Random seed resolution
    if seed < 0:
      seed = random.randint(0, 2**32 - 1)
    print(f"Executing dual-stage ComfyUI job with seed: {seed}")

    formatted_lyrics = lyrics.replace("\\n", "\n").strip()

    # 2. Load workflow_api.json
    with open("workflow_api.json", "r", encoding="utf-8") as f:
      prompt = json.load(f)

    # 3. Inject inputs into Node 23 (YuE2GenerateABC)
    if "23" in prompt:
      prompt["23"]["inputs"]["style"] = style
      prompt["23"]["inputs"]["lyrics"] = formatted_lyrics
      prompt["23"]["inputs"]["mode"] = cot
      prompt["23"]["inputs"]["max_abc_tokens"] = max_abc_tokens
      prompt["23"]["inputs"]["temperature"] = 0.70
      prompt["23"]["inputs"]["top_p"] = 0.90
      prompt["23"]["inputs"]["top_k"] = 30
      prompt["23"]["inputs"]["repetition_penalty"] = 1.005
      prompt["23"]["inputs"]["penalty_window"] = 100
      prompt["23"]["inputs"]["seed"] = seed

    # 4. Inject inputs into Node 22 (YuE2GenerateMusic)
    if "22" in prompt:
      prompt["22"]["inputs"]["style"] = style
      prompt["22"]["inputs"]["lyrics"] = formatted_lyrics
      prompt["22"]["inputs"]["mode"] = cot
      prompt["22"]["inputs"]["max_duration"] = max_duration
      prompt["22"]["inputs"]["temperature"] = temperature
      prompt["22"]["inputs"]["top_p"] = 0.95
      prompt["22"]["inputs"]["top_k"] = 100
      prompt["22"]["inputs"]["repetition_penalty"] = 1.20
      prompt["22"]["inputs"]["cfg_scale"] = 1.00
      prompt["22"]["inputs"]["seed"] = seed

      # If custom_abc is supplied, disconnect Node 23 and use raw text
      if custom_abc and custom_abc.strip():
        prompt["22"]["inputs"]["abc"] = custom_abc.strip()
      else:
        prompt["22"]["inputs"]["abc"] = ["23", 0]

    # 5. Inject inputs into Node 8 (KSampler)
    if "8" in prompt:
      prompt["8"]["inputs"]["steps"] = steps
      prompt["8"]["inputs"]["sampler_name"] = sampler_name
      prompt["8"]["inputs"]["scheduler"] = scheduler
      prompt["8"]["inputs"]["seed"] = seed
      prompt["8"]["inputs"]["cfg"] = 1.0
      prompt["8"]["inputs"]["denoise"] = 1.0

    # 6. Submit workflow to local ComfyUI instance
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")

    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=data)
    response = json.loads(urllib.request.urlopen(req).read())
    prompt_id = response["prompt_id"]

    # 7. Wait for execution to finish via WebSocket
    while True:
      out = ws.recv()
      if isinstance(out, str):
        message = json.loads(out)
        if message["type"] == "executing":
          data = message["data"]
          if data["node"] is None and data["prompt_id"] == prompt_id:
            break
      else:
        continue
    ws.close()

    # 8. Locate generated audio file recursively inside ComfyUI output directory
    output_root = "/root/ComfyUI/output"
    found_audio = []
    found_text = []

    for root, _, files in os.walk(output_root):
      for f in files:
        full_path = os.path.join(root, f)
        if f.endswith((".flac", ".wav", ".mp3")):
          found_audio.append(full_path)
        elif f.endswith((".abc", ".txt")):
          found_text.append(full_path)

    if not found_audio:
      raise RuntimeError(
          "No audio file was produced in the ComfyUI output directory."
      )

    latest_audio = max(found_audio, key=os.path.getmtime)

    abc_text = ""
    if found_text:
      latest_text = max(found_text, key=os.path.getmtime)
      with open(latest_text, "r", encoding="utf-8") as f:
        abc_text = f.read()

    # 9. Transcode to requested format
    final_output = latest_audio
    output_dir = os.path.dirname(latest_audio)

    if audio_format == "mp3" and not latest_audio.endswith(".mp3"):
      final_output = os.path.join(output_dir, f"song_{prompt_id}.mp3")
      subprocess.run(
          ["ffmpeg", "-y", "-i", latest_audio, "-b:a", "320k", final_output],
          check=True,
      )
    elif audio_format == "wav" and not latest_audio.endswith(".wav"):
      final_output = os.path.join(output_dir, f"song_{prompt_id}.wav")
      subprocess.run(
          ["ffmpeg", "-y", "-i", latest_audio, final_output], check=True
      )

    return Output(audio=Path(final_output), score_abc=abc_text)