FROM python:3.12-slim

# libopus0 = the .so opuslib loads for voice encode/decode on Linux
# (opuslib-next-bundled only ships Windows DLLs — without this,
# discord-native-voice can't import and the stock client can't encode).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libopus0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# native-voice pins discord.py-self>=2.2.0 vs master's 2.2.0a0 — needs its
# own --no-deps pass
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps discord-native-voice

COPY . .
CMD ["python", "run.py"]
