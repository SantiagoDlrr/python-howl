
"""
main.py
------------------------------------------------------------------
FastAPI server for:
  • audio upload → transcription (Gemini or WhisperX)
  • sentiment    → OCI Language
  • call report  → Gemini
  • chat         → Gemini
  • runtime settings (/settings) – defaults to WhisperX.
"""

import os
import sys
import json
import uuid
import gc
import logging
from datetime import date
from enum import Enum
from functools import lru_cache
from typing import List, Dict, Any, Optional
from datetime import date, datetime          # ← already had `date`, add `datetime`
from mutagen import File as MutagenFile     
# import torch
import uvicorn
import google.generativeai as genai
import oci
from oci.config import from_file
from oci.ai_language import AIServiceLanguageClient
from oci.ai_language.models import (
    BatchDetectLanguageSentimentsDetails, TextDocument,
)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# from transcription_module import AudioTranscriber
from temp_storage import save_transcript, load_transcript   # optional utility

# --------------------------------------------------------------------------- #
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# --------------------------------------------------------------------------- #
# FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# --------------------------------------------------------------------------- #
# Runtime settings – default **Gemini**
class TranscriptionEngine(str, Enum):
    whisperx = "whisperx"
    gemini = "gemini"

class RuntimeSettings(BaseModel):
    transcription_engine: TranscriptionEngine = TranscriptionEngine.gemini
    llm_model: str = "gemini-1.5-flash"

runtime_settings = RuntimeSettings()

@app.post("/settings", response_model=RuntimeSettings)
def update_runtime_settings(payload: RuntimeSettings):
    runtime_settings.transcription_engine = payload.transcription_engine
    runtime_settings.llm_model = payload.llm_model
    logger.info(f"Settings updated → {runtime_settings.dict()}")
    return runtime_settings

@app.get("/settings", response_model=RuntimeSettings)
def read_runtime_settings():
    return runtime_settings

# --------------------------------------------------------------------------- #
# Gemini helper: cached model factory
@lru_cache(maxsize=8)
def get_gemini_model(model_name: str):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(model_name)


# ---------------------------------------------------------------------------
# Gemini audio helper  

import os, json, logging
from typing import List, Dict, Any
import google.generativeai as genai
from google.generativeai import GenerationConfig    

logger = logging.getLogger(__name__)

SYS_PROMPT = (
    "You are a speech‑to‑text engine. Return ONLY valid JSON of the form\n"
    '{ "segments": ['
    '{"speaker":"SPEAKER_00","text":"…","start":0.123,"end":0.456}'
    "] }\n"
    "• Use SPEAKER_00, SPEAKER_01… (or real names if you can)\n"
    "• start/end are seconds from 0.0\n"
    "• No prose, markdown, or code fences."
)

def transcribe_with_gemini(audio_path: str, model_name: str) -> List[Dict[str, Any]]:
    # ---- 1. build the model with system_instruction + JSON mode -----------
    model = genai.GenerativeModel(
        model_name,
        system_instruction=SYS_PROMPT,                    # 👈 system prompt
        generation_config=GenerationConfig(
            response_mime_type="application/json"         # 👈 JSON‑only
        ),
    )

    # ---- 2. wrap the audio + mandatory text in ONE user message ----------
    ext = os.path.splitext(audio_path)[1].lower()
    mime = {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
        ".flac": "audio/flac", ".ogg": "audio/ogg",
    }.get(ext, "application/octet-stream")

    with open(audio_path, "rb") as f:
        audio_blob = {"inline_data": {"mime_type": mime, "data": f.read()}}

    prompt = [
        "Transcribe this call to JSON.",   # <- required text part&#8203;:contentReference[oaicite:2]{index=2}
        audio_blob,
    ]

    # ---- 3. call the model ------------------------------------------------
    resp = model.generate_content(prompt)

    # ---- 4. normalise -----------------------------------------------------
    try:
        data = json.loads(resp.text)                       # clean JSON
        return [
            {
                "speaker": seg["speaker"],
                "text": seg["text"].strip(),
                "start": round(float(seg["start"]), 3),
                "end":   round(float(seg["end"]),   3),
            }
            for seg in data["segments"]
        ]
    except Exception as e:
        logger.error("Gemini JSON parse failed", exc_info=True)
        return [{
            "speaker": "SYSTEM",
            "text": f"[Gemini transcription failed: {e}]",
            "start": 0.0, "end": 0.0,
        }]



# --------------------------------------------------------------------------- #
# OCI Language client (optional)
language_client: Optional[AIServiceLanguageClient] = None
try:
    if os.getenv("OCI_RESOURCE_PRINCIPAL_VERSION"):
        signer = oci.auth.signers.get_resource_principals_signer()
        language_client = AIServiceLanguageClient(config={}, signer=signer)
    else:
        oci_conf = from_file()
        language_client = AIServiceLanguageClient(oci_conf)
    logger.info("OCI Language client ready.")
except Exception:
    logger.error("OCI init failed – sentiment disabled.", exc_info=True)
    language_client = None

# --------------------------------------------------------------------------- #
# WhisperX transcriber (global – may remain unused if Gemini chosen)
# try:
#     whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
#     whisper_compute = "float16" if whisper_device == "cuda" else "int8"
#     whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
#     transcriber = AudioTranscriber(
#         model_name=whisper_model_size,
#         device=whisper_device,
#         compute_type=whisper_compute,
#     )
# except Exception:
#     logger.error("WhisperX init failed.", exc_info=True)
#     transcriber = None

# --------------------------------------------------------------------------- #
@app.get("/")
def root():
    return {"message": "Audio Analysis API ready."}

# --------------------------------------------------------------------------- #
# Helpers for LLM report
def create_llm_prompt(text: str) -> str:
    return f"""
    You are an expert call analyst. Produce JSON with:
    {{
    "feedback": "...",
    "keyTopics": [...],
    "emotions": [...],
    "sentiment": "...",
    "output": "...",
    "riskWords": "...",
    "summary": "...",
    "rating": "..."
    }}
    Transcript:
    --- START ---
    {text}
    --- END ---
    Note: rating is on a scale of 0-100
    """

def get_default_report():
    keys = ["feedback", "keyTopics", "emotions",
            "sentiment", "output", "riskWords", "summary", "rating"]
    return {k: f"Error generating {k}." for k in keys}

def generate_report(text: str) -> Dict[str, Any]:
    default = get_default_report()
    try:
        model = get_gemini_model(runtime_settings.llm_model)
    except Exception as e:
        logger.error("Gemini model init failed", exc_info=True)
        return default
    try:
        resp = model.generate_content(create_llm_prompt(text))
        content = resp.text.strip() if resp.text else ""
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        data = json.loads(content)
        for k in default:
            data.setdefault(k, default[k])
        return data
    except Exception:
        logger.error("Report generation failed", exc_info=True)
        return default

def extract_audio_metadata(path: str) -> tuple[Optional[str], Optional[float]]:
    """
    Returns (recording_date_str, duration_seconds) where either item may be None.
    • recording_date_str: original “date recorded” tag if present, else None
      (format dd/mm/yyyy for convenience)
    • duration_seconds : float with total length
    """
    try:
        audio = MutagenFile(path)
        if not audio:
            return None, None

        # -------- duration --------
        duration = getattr(audio.info, "length", None)      # seconds (float)

        # -------- date recorded --- (common tag names across formats)
        tag_keys = ("TDRC", "©day", "date", "YEAR", "TYER", "TDAT")
        rec_date = None
        if audio.tags:
            for k in tag_keys:
                if k in audio.tags:
                    raw = str(audio.tags[k])
                    # strip non‑digits, then try ISO‑like reconstruction
                    raw = raw.replace(":", "-").replace("/", "-")
                    try:
                        d = datetime.fromisoformat(raw[:10])
                        rec_date = d.strftime("%d/%m/%Y")
                    except Exception:
                        rec_date = raw        # fallback: raw tag value
                    break

        return rec_date, duration
    except Exception as e:
        logger.warning(f"extract_audio_metadata failed: {e}")
        return None, None



# --------------------------------------------------------------------------- #
# ----------------------------- /upload ------------------------------------- #
@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file received.")
    # if transcriber is None and runtime_settings.transcription_engine == TranscriptionEngine.whisperx:
    #     raise HTTPException(status_code=503, detail="WhisperX unavailable.")

    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"audio_{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file.")
        with open(temp_path, "wb") as f:
            f.write(data)
    except Exception as e:
        logger.error("File save failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    # ----------------- metadata (date recorded & duration) --------------------
    rec_date, duration_sec = extract_audio_metadata(temp_path)


    # • Fallback to file mtime if the audio file has no date tag
    if not rec_date:
        rec_date = datetime.fromtimestamp(
            os.path.getmtime(temp_path)
        ).strftime("%d/%m/%Y")

    # • Friendly HH:MM:SS string (or "N/A")
    def fmt_dur(s: Optional[float]) -> str:
        if s is None:
            return "N/A"
        h, m = divmod(int(s), 3600)
        m, s = divmod(m, 60)
        return f"{h:02}:{m:02}:{s:02}"

    duration_str = fmt_dur(duration_sec)




    # Transcription
    try:
        if runtime_settings.transcription_engine == TranscriptionEngine.gemini:
            diarized = transcribe_with_gemini(temp_path, runtime_settings.llm_model)
        # else:
        #     diarized = transcriber.transcribe_and_diarize(temp_path)
    except Exception as e:
        logger.error("Transcription failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    full_text = "\n".join(f"{d['speaker']}: {d['text']}" for d in diarized)

    # OCI Sentiment
    oci_emotion, oci_aspects = "N/A", []
    if language_client and full_text:
        try:
            docs = [TextDocument(key="1", text=full_text, language_code="en")]
            req = BatchDetectLanguageSentimentsDetails(documents=docs)
            resp = language_client.batch_detect_language_sentiments(req)
            if resp.data.documents:
                doc = resp.data.documents[0]
                oci_emotion = getattr(doc.document_sentiment, "label", "Neutral")
                oci_aspects = [
                    {"text": a.text, "sentiment": a.sentiment, "scores": a.scores}
                    for a in (doc.aspects or [])
                ]
        except Exception:
            logger.error("OCI sentiment failed", exc_info=True)

    # LLM report
    report = generate_report(full_text)

    # Save (optional)
    transcript_id = str(uuid.uuid4())
    try:
        save_transcript(
            transcript_id=transcript_id,
            diarized_transcript=diarized,
            full_transcript_text=full_text,
            report_data=report,
            oci_emotion=oci_emotion,
            oci_aspects=oci_aspects,
            recordingDate=rec_date,          # ★ persists to JSON on disk
            duration=duration_str,           #   (keep both names & types
            durationSeconds=duration_sec, 
        )
    except Exception as e:
        logger.warning(f"save_transcript failed: {e}")

    # Clean up
    try:
        os.remove(temp_path)
    except OSError:
        pass

    return {
        "id": transcript_id,
        "name": file.filename,
        "date": rec_date,
        "type": "Sin categoría",
        "duration": duration_str,
        "rating": 0,
        "report": report,
        "transcript": diarized,
    }

# --------------------------------------------------------------------------- #
# ------------------------------ /chat -------------------------------------- #
class ChatRequest(BaseModel):
    transcript_id: str
    messages: List[Dict[str, str]]

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        data = load_transcript(req.transcript_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Transcript not found.")
    except Exception:
        logger.error("load_transcript error", exc_info=True)
        raise HTTPException(status_code=500, detail="Server error.")

    transcript = data.get("full_transcript_text") or "\n".join(
        f"{d['speaker']}: {d['text']}" for d in data.get("diarized_transcript", [])
    )
    rpt = data.get("report_data", {})
    oci_emotion = data.get("oci_emotion", "N/A")

    context = "\n".join([
        "You are a helpful assistant analysing a recorded call.",
        "--- Report ---",
        f"Feedback: {rpt.get('feedback','')}",
        f"KeyTopics: {', '.join(rpt.get('keyTopics', []))}",
        f"Emotions: {', '.join(rpt.get('emotions', []))}",
        f"Sentiment: {rpt.get('sentiment','')}",
        f"Output: {rpt.get('output','')}",
        f"RiskWords: {rpt.get('riskWords','')}",
        f"Summary: {rpt.get('summary','')}",
        f"OCI sentiment: {oci_emotion}",
        "--- Transcript ---",
        transcript,
    ])

    messages = [
        {"role": "user", "parts": [context]},
        {"role": "model", "parts": ["OK – how can I assist?"]},
    ] + [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in req.messages
    ]

    try:
        model = get_gemini_model(runtime_settings.llm_model)
        resp = model.generate_content(messages)
        answer = resp.text.strip() if resp.text else ""
        return {"assistant_message": answer or "Sorry, I have no reply."}
    except Exception:
        logger.error("Gemini chat failed", exc_info=True)
        raise HTTPException(status_code=503, detail="LLM service error.")

# --------------------------------------------------------------------------- #
# Dev utilities
@app.get("/test_oci")
def test_oci():
    if not language_client:
        return {"error": "OCI not configured"}
    sample = "I am unhappy with the service but the agent was helpful."
    docs = [TextDocument(key="1", text=sample, language_code="en")]
    req = BatchDetectLanguageSentimentsDetails(documents=docs)
    resp = language_client.batch_detect_language_sentiments(req)
    return {"data": str(resp.data)}

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY not set – LLM endpoints will fail.")
    # if transcriber is None:
    #     print("WARNING: WhisperX unavailable – only Gemini transcription will work.")
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set – diarization will fail with WhisperX.")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")





















