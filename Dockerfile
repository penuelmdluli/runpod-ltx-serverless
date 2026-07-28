# RunPod serverless LTX-Video worker — 13B 0.9.7 DISTILLED (high quality).
# SLIM image: the model is NOT baked (that made a 33GB image that failed to pull
# with "unexpected EOF"). Instead the weights download at first run to the
# attached network volume (mounted at /runpod-volume) and persist across cold
# starts. Image stays ~6GB and pulls reliably.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# HF cache lives on the network volume so the ~35GB download happens once.
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_OFFLINE=0

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Let git-diffusers resolve its own transformers/huggingface_hub/accelerate
# versions (pinning an old huggingface_hub conflicts with diffusers-from-git).
RUN pip install --no-cache-dir \
      "git+https://github.com/huggingface/diffusers.git" \
      transformers accelerate sentencepiece tiktoken protobuf \
      imageio imageio-ffmpeg \
      "runpod==1.7.7" Pillow requests

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
