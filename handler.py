"""RunPod Serverless handler — LTX-Video 0.9.7 13B DISTILLED, MAX QUALITY.

Two-stage spatial-upscaler flow (Lightricks' recommended best path):
  1) generate at downscaled res  ->  2) latent upscale 2x  ->  3) refine  ->  4) downscale.
T2V + I2V.

Input (event["input"]): prompt (req), image (opt -> I2V), width/height (target,
default 1216x704), num_frames (default 97), fps (24), seed.
Output: { video_base64, mp4_bytes, num_frames, fps, kind, total_seconds } | { error, trace }
"""
import base64
import io
import os
import tempfile
import time
import traceback

import runpod

MODEL = "Lightricks/LTX-Video-0.9.7-distilled"
UPSAMPLER = "Lightricks/ltxv-spatial-upscaler-0.9.7"
_P = {}


def _log(m):
    print(f"[handler] {m}", flush=True)


def _pipes():
    if "pipe" in _P:
        return _P["pipe"], _P["up"]
    import torch
    from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
    t = time.time()
    pipe = LTXConditionPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    up = LTXLatentUpsamplePipeline.from_pretrained(UPSAMPLER, vae=pipe.vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); up.to("cuda")
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    _P["pipe"], _P["up"] = pipe, up
    _log(f"loaded pipelines in {time.time()-t:.1f}s")
    return pipe, up


def _decode_image(field):
    from PIL import Image
    field = str(field).strip()
    if field.startswith(("http://", "https://")):
        import requests
        data = requests.get(field, timeout=120).content
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
        neg = inp.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted")
        ew = int(inp.get("width", 1216)); eh = int(inp.get("height", 704))
        num_frames = int(inp.get("num_frames", 97))
        if (num_frames - 1) % 8 != 0:
            num_frames = ((num_frames - 1) // 8) * 8 + 1
        fps = int(inp.get("fps", 24))
        seed = int(inp.get("seed", 0))
        image_field = inp.get("image")

        pipe, up = _pipes()
        ratio = pipe.vae_spatial_compression_ratio

        # stage-1 downscaled resolution (2/3), rounded to VAE-acceptable
        dw = int(ew * 2 / 3); dh = int(eh * 2 / 3)
        dw -= dw % ratio; dh -= dh % ratio

        conditions = None
        kind = "t2v"
        if image_field:
            from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
            img = _decode_image(image_field).resize((dw, dh))
            conditions = [LTXVideoCondition(video=[img], frame_index=0)]
            kind = "i2v"

        base = dict(prompt=prompt, negative_prompt=neg, num_frames=num_frames,
                    decode_timestep=0.05, decode_noise_scale=0.025, image_cond_noise_scale=0.0,
                    guidance_scale=1.0, guidance_rescale=0.7)
        if conditions:
            base["conditions"] = conditions

        _log(f"stage1 {kind} {dw}x{dh} frames={num_frames}")
        latents = pipe(width=dw, height=dh, timesteps=[1000, 993, 987, 981, 975, 909, 725, 0.03],
                       output_type="latent", generator=torch.Generator().manual_seed(seed), **base).frames

        _log("stage2 upscale 2x")
        up_latents = up(latents=latents, adain_factor=1.0, output_type="latent").frames
        uw, uh = dw * 2, dh * 2

        _log(f"stage3 refine {uw}x{uh}")
        video = pipe(width=uw, height=uh, denoise_strength=0.4, timesteps=[1000, 909, 725, 421, 0],
                     latents=up_latents, output_type="pil",
                     generator=torch.Generator().manual_seed(seed), **base).frames[0]
        video = [f.resize((ew, eh)) for f in video]

        out_path = os.path.join(tempfile.gettempdir(), f"out_{int(t0)}.mp4")
        export_to_video(video, out_path, fps=fps)
        size = os.path.getsize(out_path)
        with open(out_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return {"video_base64": b64, "mp4_bytes": size, "num_frames": len(video),
                "fps": fps, "kind": kind, "total_seconds": round(time.time() - t0, 1)}
    except Exception as e:
        tb = traceback.format_exc()
        _log("ERROR:\n" + tb)
        return {"error": str(e), "trace": tb[-1800:]}


runpod.serverless.start({"handler": handler})
