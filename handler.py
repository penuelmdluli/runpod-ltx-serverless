"""RunPod Serverless handler — LTX-Video 0.9.7 13B DISTILLED (T2V + I2V).

Input (event["input"]):
  prompt          (str, required)
  image           (str, optional) base64 / data-URI / http(s) URL -> Image-to-Video
  negative_prompt (str)
  width, height   (int)   default 1216 x 704 (rounded to /32)
  num_frames      (int)   default 97  (rounded to 8n+1)
  steps           (int)   default 8   (distilled: 4-10)
  fps             (int)   default 24
  seed            (int)   optional

Output: { video_base64, mp4_bytes, num_frames, fps, kind, total_seconds } | { error, trace }
"""
import base64
import io
import os
import time
import traceback
import tempfile

import runpod

MODEL = os.environ.get("LTX_MODEL", "Lightricks/LTX-Video-0.9.7-distilled")
MAX_B64 = int(os.environ.get("MAX_B64_BYTES", "18000000"))
_PIPE = {}


def _log(m):
    print(f"[handler] {m}", flush=True)


def _get_pipe():
    if "p" in _PIPE:
        return _PIPE["p"]
    import torch
    from diffusers import LTXConditionPipeline
    t = time.time()
    pipe = LTXConditionPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()          # fit the 13B on a 48GB card
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    _PIPE["p"] = pipe
    _log(f"loaded LTXConditionPipeline in {time.time()-t:.1f}s")
    return pipe


def _decode_image(field):
    from PIL import Image
    field = field.strip()
    if field.startswith(("http://", "https://")):
        import requests
        data = requests.get(field, timeout=60).content
    else:
        if field.startswith("data:") and "," in field:
            field = field.split(",", 1)[1]
        data = base64.b64decode(field)
    return Image.open(io.BytesIO(data)).convert("RGB")


def handler(event):
    t0 = time.time()
    try:
        import torch
        from diffusers.utils import export_to_video
        inp = (event or {}).get("input", {}) or {}
        prompt = inp.get("prompt")
        if not prompt or not str(prompt).strip():
            return {"error": "missing required field 'prompt'"}

        neg = inp.get("negative_prompt",
                      "worst quality, inconsistent motion, blurry, jittery, distorted")
        width = int(inp.get("width", 1216))
        height = int(inp.get("height", 704))
        num_frames = int(inp.get("num_frames", 97))
        steps = int(inp.get("steps", 8))
        fps = int(inp.get("fps", 24))
        seed = inp.get("seed", None)
        image_field = inp.get("image")

        width -= width % 32
        height -= height % 32
        if (num_frames - 1) % 8 != 0:
            num_frames = ((num_frames - 1) // 8) * 8 + 1

        gen = None
        if seed is not None:
            gen = torch.Generator(device="cuda").manual_seed(int(seed))

        pipe = _get_pipe()
        kwargs = dict(
            prompt=prompt, negative_prompt=neg, width=width, height=height,
            num_frames=num_frames, num_inference_steps=steps,
            guidance_scale=1.0,                # distilled -> 1.0
            decode_timestep=0.05, decode_noise_scale=0.025, image_cond_noise_scale=0.0,
            generator=gen,
        )

        kind = "t2v"
        if image_field:
            from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
            img = _decode_image(image_field).resize((width, height))
            kwargs["conditions"] = [LTXVideoCondition(video=[img], frame_index=0)]
            kind = "i2v"

        _log(f"gen {kind} {width}x{height} frames={num_frames} steps={steps}")
        ti = time.time()
        frames = pipe(**kwargs).frames[0]
        _log(f"inference done in {time.time()-ti:.1f}s ({len(frames)} frames)")

        out_path = os.path.join(tempfile.gettempdir(), f"out_{int(t0)}.mp4")
        export_to_video(frames, out_path, fps=fps)
        size = os.path.getsize(out_path)
        with open(out_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        try:
            os.remove(out_path)
        except OSError:
            pass
        if len(b64) > MAX_B64:
            return {"error": f"video base64 {len(b64)} exceeds MAX_B64 {MAX_B64}"}

        return {"video_base64": b64, "mp4_bytes": size, "num_frames": len(frames),
                "fps": fps, "kind": kind, "total_seconds": round(time.time() - t0, 1)}
    except Exception as e:
        tb = traceback.format_exc()
        _log("ERROR:\n" + tb)
        return {"error": str(e), "trace": tb[-1800:]}


runpod.serverless.start({"handler": handler})
