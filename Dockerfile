# RunPod serverless LTX-Video worker — 13B 0.9.7 DISTILLED (high quality).
# Weights baked at build. LTXConditionPipeline needs recent diffusers (0.9.7
# support), so we install diffusers from git.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/models HF_HUB_OFFLINE=0

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      "git+https://github.com/huggingface/diffusers.git" \
      "transformers==4.49.0" \
      "accelerate==1.4.0" \
      "sentencepiece==0.2.0" \
      "imageio==2.36.1" "imageio-ffmpeg==0.5.1" \
      "huggingface_hub==0.27.1" \
      "runpod==1.7.7" "Pillow==11.0.0" "requests==2.32.3"

# Bake the 13B distilled model (validates diffusers<->weights at build time).
RUN python -c "import torch; from diffusers import LTXConditionPipeline; \
LTXConditionPipeline.from_pretrained('Lightricks/LTX-Video-0.9.7-distilled', torch_dtype=torch.bfloat16); \
print('LTX 0.9.7 distilled baked OK')"

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
