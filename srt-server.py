import tempfile
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from loguru import logger

app = FastAPI()

# Using 'tiny.en' on GPU. It's tiny, fast, and uses almost no VRAM.
logger.info("Loading Tiny-Whisper onto GPU...")
engine = WhisperModel("tiny.en", device="cuda", compute_type="float16")

@app.post("/inference")
async def inference(file: UploadFile = File(...), temperature: float = Form(0.0)):
    try:
        suffix = os.path.splitext(file.filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        segments, _ = engine.transcribe(tmp_path, beam_size=1)
        text = "".join(segment.text for segment in segments)
        
        os.remove(tmp_path)
        logger.info(f"Heard: {text}")
        return JSONResponse(content={"text": text.strip()})
    except Exception as e:
        return JSONResponse(content={"text": "", "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

