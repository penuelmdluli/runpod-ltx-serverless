# RunPod serverless LTX-Video worker — 13B 0.9.7 DISTILLED (high quality).
# Weights baked at build. LTXConditionPipeline needs recent diffusers (0.9.7
# support), so we install diffusers from git.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/models HF_HUB_OFFLINE=0

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Let git-diffusers resolve its own transformers/huggingface_hub/accelerate
# versions (pinning an old huggingface_hub conflicts with diffusers-from-git).
RUN pip install --no-cache-dir \
      "git+https://github.com/huggingface/diffusers.git" \
      transformers accelerate sentencepiece tiktoken protobuf \
      imageio imageio-ffmpeg \
      "runpod==1.7.7" Pillow requests

# Bake the 13B distilled model + the spatial upscaler (2-stage max-quality flow).
RUN python -c "import torch; from diffusers import LTXConditionPipeline; from huggingface_hub import snapshot_download; \
LTXConditionPipeline.from_pretrained('Lightricks/LTX-Video-0.9.7-distilled', torch_dtype=torch.bfloat16); \
snapshot_download('Lightricks/ltxv-spatial-upscaler-0.9.7'); \
print('LTX 0.9.7 + upscaler baked OK')"

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
