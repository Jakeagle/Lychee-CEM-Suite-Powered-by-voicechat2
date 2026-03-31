import io
import torch
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from TTS.api import TTS
from loguru import logger

app = FastAPI()

# VITS is instant on your Ryzen 3900X
MODEL_NAME = "tts_models/en/vctk/vits"

logger.info("Loading VITS Voice (CPU) with Resampling...")
# 'p273' is a great dry, sarcastic voice for Bones
tts = TTS(MODEL_NAME).to("cpu")

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        # 1. Generate audio at native 22050Hz
        wav = tts.tts(text=request.text, speaker="p335")
        wav_np = np.array(wav)

        # 2. Resample from 22050Hz to 24000Hz (Opus requirement)
        # This uses your 3900X's power to fix the sample rate in real-time
        wav_24k = librosa.resample(wav_np, orig_sr=22050, target_sr=24000)

        # 3. Clip audio to prevent "popping" or distortion
        wav_24k = np.clip(wav_24k, -1.0, 1.0)

        # 4. Write to buffer as Ogg/Opus at the correct rate
        buffer = io.BytesIO()
        sf.write(buffer, wav_24k, 24000, format='ogg', subtype='opus')
        buffer.seek(0)
        
        logger.info(f"Synthesized and Resampled: {request.text[:30]}...")
        return StreamingResponse(buffer, media_type="audio/ogg; codecs=opus")
        
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
