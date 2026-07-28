"""RunPod Serverless handler — LTX-Video T2V + I2V.

Input (event["input"]):
  prompt          (str, required)
  image           (str, optional) base64 (optionally data: URI) OR http(s) URL.
                  If present -> Image-to-Video, else Text-to-Video.
  negative_prompt (str)   default a generic quality negative
  width, height   (int)   default 704 x 480 (must be multiples of 32)
  num_frames      (int)   default 121 (must be 8*n + 1); ~5s at 24 fps
  steps           (int)   default 40
  fps             (int)   default 24
  seed            (int)   optional, for reproducibility

Output:
  { video_base64, mp4_bytes, num_frames, fps, kind, seconds }   on success
  { error, trace }                                              on failure
"""
import base64
import io
import os
import time
import traceback
import tempfile

import runpod

MODEL = os.environ.get("LTX_MODEL", "Lightricks/LTX-Video")
# RunPod stores async job output; keep base64 comfortably under the limit.
MAX_B64 = int(os.environ.get("MAX_B64_BYTES", "18000000"))  # ~18 MB

_PIPES = {}


def _log(msg):
    print(f"[handler] {msg}", flush=True)


def _vram(tag):
    try:
        import torch
        if torch.cuda.is_available():
            _log(f"VRAM[{tag}] allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
                 f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")
    except Exception:
        pass


def _get_pipe(kind):
    """Lazy-load and cache the T2V or I2V pipeline. I2V reuses T2V weights."""
    import torch
    if kind in _PIPES:
        return _PIPES[kind]
    _vram(f"before_load_{kind}")
    t = time.time()
    if kind == "t2v":
        from diffusers import LTXPipeline
        pipe = LTXPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
    else:
        from diffusers import LTXImageToVideoPipeline
        if "t2v" in _PIPES:
            # Share the already-loaded (and already-offloaded) components.
            pipe = LTXImageToVideoPipeline.from_pipe(_PIPES["t2v"])
        else:
            pipe = LTXImageToVideoPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
            pipe.enable_model_cpu_offload()
    _PIPES[kind] = pipe
    _log(f"loaded {kind} pipeline in {time.time()-t:.1f}s")
    _vram(f"after_load_{kind}")
    return pipe


def _decode_image(field):
    from PIL import Image
    field = field.strip()
    if field.startswith("http://") or field.startswith("https://"):
        import requests
        r = requests.get(field, timeout=60)
        r.raise_for_status()
        data = r.content
    else:
        if field.startswith("data:") and "," in field:
            field = field.split(",", 1)[1]
        data = base64.b64decode(field)
    return Image.open(io.BytesIO(data)).convert("RGB")


def handler(event):
    t0 = time.time()
    try:
        inp = (event or {}).get("input", {}) or {}
        prompt = inp.get("prompt")
        if not prompt or not str(prompt).strip():
            return {"error": "missing required field 'prompt'"}

        neg = inp.get("negative_prompt",
                      "worst quality, blurry, distorted, deformed, watermark, text, static, jitter")
        width = int(inp.get("width", 704))
        height = int(inp.get("height", 480))
        num_frames = int(inp.get("num_frames", 121))
        steps = int(inp.get("steps", 40))
        fps = int(inp.get("fps", 24))
        seed = inp.get("seed", None)
        image_field = inp.get("image")

        # LTX constraints: width/height multiple of 32; num_frames = 8k + 1.
        width -= width % 32
        height -= height % 32
        if (num_frames - 1) % 8 != 0:
            num_frames = ((num_frames - 1) // 8) * 8 + 1

        import torch
        gen = None
        if seed is not None:
            gen = torch.Generator(device="cuda").manual_seed(int(seed))

        kind = "i2v" if image_field else "t2v"
        _log(f"request kind={kind} {width}x{height} frames={num_frames} steps={steps} "
             f"seed={seed} prompt={str(prompt)[:70]!r}")

        pipe = _get_pipe(kind)
        kwargs = dict(prompt=prompt, negative_prompt=neg, width=width, height=height,
                      num_frames=num_frames, num_inference_steps=steps, generator=gen)
        if kind == "i2v":
            img = _decode_image(image_field).resize((width, height))
            kwargs["image"] = img
            _log("input image decoded + resized")

        _log("inference start")
        ti = time.time()
        frames = pipe(**kwargs).frames[0]
        infer_s = time.time() - ti
        _log(f"inference done in {infer_s:.1f}s ({len(frames)} frames)")
        _vram("after_infer")

        from diffusers.utils import export_to_video
        out_path = os.path.join(tempfile.gettempdir(), f"out_{int(t0)}.mp4")
        export_to_video(frames, out_path, fps=fps)
        size = os.path.getsize(out_path)
        _log(f"encoded {out_path} ({size/1e6:.2f} MB)")

        with open(out_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        try:
            os.remove(out_path)
        except OSError:
            pass

        if len(b64) > MAX_B64:
            return {"error": f"video base64 {len(b64)} exceeds MAX_B64 {MAX_B64}; "
                             f"lower num_frames/resolution or enable object storage"}

        return {
            "video_base64": b64,
            "mp4_bytes": size,
            "num_frames": len(frames),
            "fps": fps,
            "kind": kind,
            "infer_seconds": round(infer_s, 1),
            "total_seconds": round(time.time() - t0, 1),
        }
    except Exception as e:
        tb = traceback.format_exc()
        _log("ERROR:\n" + tb)
        return {"error": str(e), "trace": tb[-1800:]}


runpod.serverless.start({"handler": handler})
