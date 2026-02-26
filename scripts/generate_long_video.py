#!/usr/bin/env python3
"""
generate_long_video.py — Generate 1–5 minute videos using Wan2.2 T2V via ComfyUI API.

Each clip is ~5 seconds (81 frames @ 16fps). Multiple clips are queued,
waited on, then stitched together with FFmpeg.

Usage examples:
  python generate_long_video.py \
      --subject "a female fashion model with long black hair, wearing a red evening gown" \
      --background "luxury hotel lobby with marble floors and golden lighting" \
      --prompt "walking gracefully, confident pose, cinematic 4K, smooth camera movement" \
      --duration 2 \
      --output my_fashion_video.mp4

  python generate_long_video.py \
      --subject "a male model in a tailored grey suit" \
      --background "modern city rooftop at sunset" \
      --duration 1 \
      --steps 20 \
      --output quick_test.mp4
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_OUTPUT_DIR = Path(__file__).parent.parent / "ComfyUI" / "output"

# Wan2.2 model paths (relative to ComfyUI models dir)
UNET_MODEL  = "Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model.safetensors.index.json"
CLIP_MODEL  = "models_t5_umt5-xxl-enc-bf16.pth"
VAE_MODEL   = "Wan2.1_VAE.pth"

FRAMES_PER_CLIP = 81   # = 5.0625 sec @ 16fps  (must be 4N+1: 17,33,49,65,81)
FPS             = 16
CLIP_DURATION   = FRAMES_PER_CLIP / FPS   # ~5.06 seconds

DEFAULT_NEG = (
    "blurry, low quality, watermark, text overlay, distorted, "
    "static, noisy, artifacts, overexposed, underexposed, "
    "bad anatomy, deformed face, extra limbs"
)

WIDTH  = 832
HEIGHT = 480
# ───────────────────────────────────────────────────────────────────────────────


def build_prompt_text(subject: str, background: str, extra: str, clip_index: int, total_clips: int) -> str:
    """Build a cinematic prompt for each clip, varying slightly to avoid repetition."""
    camera_moves = [
        "slow dolly forward",
        "gentle pan left to right",
        "subtle zoom out",
        "static wide shot",
        "smooth tracking shot",
        "slow push in",
        "elegant crane shot moving down",
        "gentle orbit around subject",
    ]
    lighting_notes = [
        "soft natural light",
        "golden hour lighting",
        "dramatic side lighting",
        "soft diffused light",
        "warm ambient glow",
    ]

    move  = camera_moves[clip_index % len(camera_moves)]
    light = lighting_notes[clip_index % len(lighting_notes)]

    base = f"{subject}, {background}, {move}, {light}, cinematic 4K, photorealistic, high detail"
    if extra:
        base += f", {extra}"
    return base


def build_workflow(positive: str, negative: str, seed: int, steps: int) -> dict:
    """Build a ComfyUI API workflow dict for Wan2.2 T2V."""
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": UNET_MODEL,
                "weight_dtype": "default"
            }
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": CLIP_MODEL,
                "type": "wan"
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": positive,
                "clip": ["2", 0]
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative,
                "clip": ["2", 0]
            }
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": VAE_MODEL
            }
        },
        "6": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["3", 0],
                "negative": ["4", 0],
                "vae":      ["5", 0],
                "width":    WIDTH,
                "height":   HEIGHT,
                "length":   FRAMES_PER_CLIP,
                "batch_size": 1
            }
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model":        ["1", 0],
                "positive":     ["6", 0],
                "negative":     ["6", 1],
                "latent_image": ["6", 2],
                "seed":         seed,
                "control_after_generate": "fixed",
                "steps":        steps,
                "cfg":          6.0,
                "sampler_name": "euler",
                "scheduler":    "simple",
                "denoise":      1.0
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae":     ["5", 0]
            }
        },
        "9": {
            "class_type": "SaveWEBM",
            "inputs": {
                "images":    ["8", 0],
                "filename_prefix": "clip",
                "codec": "vp9",
                "fps": FPS,
                "crf": 32.0
            }
        }
    }


def api_post(endpoint: str, data: dict) -> dict:
    body = json.dumps(data).encode("utf-8")
    req  = urllib.request.Request(
        f"{COMFYUI_URL}/{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def api_get(endpoint: str) -> dict:
    with urllib.request.urlopen(f"{COMFYUI_URL}/{endpoint}", timeout=30) as resp:
        return json.loads(resp.read())


def check_server() -> bool:
    try:
        api_get("system_stats")
        return True
    except Exception:
        return False


def queue_prompt(workflow: dict) -> str:
    """Submit a workflow and return its prompt_id."""
    result = api_post("prompt", {"prompt": workflow})
    return result["prompt_id"]


def wait_for_completion(prompt_id: str, clip_num: int, total: int, timeout: int = 3600) -> list[str]:
    """Poll /history until the prompt completes. Returns list of output file paths."""
    print(f"  [clip {clip_num}/{total}] Generating... (this takes ~2–5 min per clip on CPU)", flush=True)
    start = time.time()
    dots  = 0
    while time.time() - start < timeout:
        try:
            history = api_get(f"history/{prompt_id}")
            if prompt_id in history:
                entry   = history[prompt_id]
                outputs = entry.get("outputs", {})
                files   = []
                for node_output in outputs.values():
                    for key in ("gifs", "images", "videos"):
                        for item in node_output.get(key, []):
                            fname = item.get("filename", "")
                            if fname:
                                files.append(str(COMFYUI_OUTPUT_DIR / fname))
                if files:
                    print(f"\r  [clip {clip_num}/{total}] Done in {time.time()-start:.0f}s          ")
                    return files
        except Exception:
            pass
        time.sleep(10)
        dots += 1
        print(f"\r  [clip {clip_num}/{total}] {'.' * (dots % 40)}", end="", flush=True)

    raise TimeoutError(f"Clip {clip_num} timed out after {timeout}s")


def concatenate_clips(clip_files: list[str], output_path: str, fps: int = FPS):
    """Use FFmpeg to concatenate all clips into a single MP4."""
    filelist = Path(output_path).with_suffix(".filelist.txt")
    with open(filelist, "w") as f:
        for path in clip_files:
            f.write(f"file '{path}'\n")

    cmd = (
        f'ffmpeg -y -f concat -safe 0 -i "{filelist}" '
        f'-c:v libx264 -preset fast -crf 18 '
        f'-pix_fmt yuv420p '
        f'"{output_path}"'
    )
    print(f"\nStitching {len(clip_files)} clips with FFmpeg...")
    ret = os.system(cmd)
    filelist.unlink(missing_ok=True)
    if ret != 0:
        raise RuntimeError("FFmpeg failed. Check the command above for details.")
    print(f"Done! Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate long videos (1–5 min) with Wan2.2 via ComfyUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--subject",    required=True,
                        help="Description of the person/model (e.g. 'female model in red dress')")
    parser.add_argument("--background", required=True,
                        help="Scene/background description (e.g. 'luxury hotel lobby')")
    parser.add_argument("--prompt",     default="",
                        help="Extra action/style details appended to each clip prompt")
    parser.add_argument("--negative",   default=DEFAULT_NEG,
                        help="Negative prompt (what to avoid)")
    parser.add_argument("--duration",   type=float, default=1.0,
                        help="Target duration in minutes (default: 1, range: 0.5–5)")
    parser.add_argument("--steps",      type=int,   default=30,
                        help="Diffusion steps per clip (default: 30; use 20 for faster/lower quality)")
    parser.add_argument("--seed",       type=int,   default=-1,
                        help="Base seed (-1 = random per clip)")
    parser.add_argument("--output",     default="output_video.mp4",
                        help="Output MP4 filename (default: output_video.mp4)")
    parser.add_argument("--comfyui-url", default=COMFYUI_URL,
                        help=f"ComfyUI server URL (default: {COMFYUI_URL})")
    args = parser.parse_args()

    global COMFYUI_URL
    COMFYUI_URL = args.comfyui_url.rstrip("/")

    # Validate duration
    if not 0.1 <= args.duration <= 10:
        print("ERROR: --duration must be between 0.1 and 10 minutes")
        sys.exit(1)

    total_seconds = args.duration * 60
    num_clips     = max(1, round(total_seconds / CLIP_DURATION))
    actual_mins   = (num_clips * CLIP_DURATION) / 60

    print("=" * 60)
    print("  Wan2.2 Long Video Generator")
    print("=" * 60)
    print(f"  Subject    : {args.subject}")
    print(f"  Background : {args.background}")
    print(f"  Duration   : {args.duration} min → {num_clips} clips ({actual_mins:.1f} min actual)")
    print(f"  Steps/clip : {args.steps}")
    print(f"  Output     : {args.output}")
    print("=" * 60)

    # Check ComfyUI is running
    print(f"\nChecking ComfyUI at {COMFYUI_URL}...")
    if not check_server():
        print("ERROR: ComfyUI is not running. Start it first:")
        print('  cd ComfyUI && python main.py --port 8188 --cpu')
        sys.exit(1)
    print("ComfyUI is online.\n")

    # Queue and wait for each clip
    clip_files = []
    base_seed  = random.randint(0, 2**32 - 1) if args.seed == -1 else args.seed

    for i in range(num_clips):
        seed     = base_seed + i
        positive = build_prompt_text(args.subject, args.background, args.prompt, i, num_clips)
        workflow = build_workflow(positive, args.negative, seed, args.steps)

        print(f"[{i+1}/{num_clips}] Prompt: {positive[:80]}...")
        try:
            prompt_id = queue_prompt(workflow)
            files     = wait_for_completion(prompt_id, i + 1, num_clips)
            # Pick the most recently modified matching file
            webm_files = [f for f in files if f.endswith(".webm")]
            if webm_files:
                clip_files.append(webm_files[0])
                print(f"  Saved: {webm_files[0]}")
            else:
                print(f"  WARNING: No .webm output found for clip {i+1}, skipping.")
        except TimeoutError as e:
            print(f"  WARNING: {e} — skipping this clip.")
        except Exception as e:
            print(f"  ERROR on clip {i+1}: {e}")
            sys.exit(1)

    if not clip_files:
        print("\nERROR: No clips were generated successfully.")
        sys.exit(1)

    print(f"\n{len(clip_files)}/{num_clips} clips generated successfully.")

    # Stitch together
    concatenate_clips(clip_files, args.output)

    actual_duration = len(clip_files) * CLIP_DURATION
    print(f"\nFinal video: {args.output}")
    print(f"Duration   : {actual_duration:.1f} seconds ({actual_duration/60:.1f} minutes)")
    print(f"Resolution : {WIDTH}x{HEIGHT} @ {FPS}fps")


if __name__ == "__main__":
    main()
