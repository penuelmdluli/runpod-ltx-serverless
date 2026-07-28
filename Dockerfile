# Standalone RunPod Serverless video worker — LTX-Video (T2V + I2V).
# Weights are BAKED IN at build time (no runtime HuggingFace download).
#
# Pinned, known-good stack (validated against Lightricks/LTX-Video
# model_index.json -> _diffusers_version 0.32.0):
#   torch 2.5.1 + CUDA 12.4 (base image)  |  diffusers 0.32.2  |  transformers 4.47.1
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_OFFLINE=0

# ffmpeg is required by imageio-ffmpeg for mp4 export.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      "diffusers==0.32.2" \
      "transformers==4.47.1" \
      "accelerate==1.2.1" \
      "sentencepiece==0.2.0" \
      "imageio==2.36.1" \
      "imageio-ffmpeg==0.5.1" \
      "huggingface_hub==0.27.1" \
      "runpod==1.7.7" \
      "Pillow==11.0.0" \
      "requests==2.32.3"

# Bake the LTX-Video weights into the image AND validate diffusers<->weights
# compatibility at build time (fails the build early if the combo is wrong).
# from_pretrained pulls ONLY the diffusers components (transformer, vae,
# text_encoder T5, tokenizer, scheduler) — not the root single-file checkpoints.
RUN python -c "import torch; from diffusers import LTXPipeline; \
p=LTXPipeline.from_pretrained('Lightricks/LTX-Video', torch_dtype=torch.bfloat16); \
print('LTX-Video weights baked OK')"

# Make the offline cache authoritative at runtime (no network needed to load).
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
