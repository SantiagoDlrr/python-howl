
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
    allow_origins=["*"],  # Allow all origins
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


CATEGORIES = ["Sales", "Technology", "HR", "Customer Support",
              "Finance", "Marketing", "Operations", "Other"]

def ensure_list(val):
    """Accept str | list | None → list[str]."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        return [x.strip() for x in val.split(",") if x.strip()]
    return []

def classify_call(text: str, model_name: str = None) -> str:
    """
    Quick 1-shot classification with Gemini.
    Returns a value from CATEGORIES (or 'Other' on failure).
    """
    try:
        model = get_gemini_model(model_name or runtime_settings.llm_model)
        prompt = (
            "Choose the single most relevant category for this call from "
            f"{CATEGORIES}. Just return the word.\n\nTranscript:\n{text}"
        )
        resp = model.generate_content(prompt)
        guess = (resp.text or "").strip().splitlines()[0]
        return guess if guess in CATEGORIES else "Other"
    except Exception:
        logger.warning("Category classification failed", exc_info=True)
        return "Other"





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
    categories = ["Sales", "Technology", "HR", "Customer Support",
                  "Finance", "Marketing", "Operations", "Other"]


    return f"""
    You are an expert call analyst. Produce JSON with:
    {{
    "feedback": "...",
    "keyTopics": [...],
    "emotions": [...],
    "sentiment": "...",
    "output": "...",
    "riskWords": [...],
    "summary": "...",
    "rating": "...",
    "category": "one value from {categories}"
    }}
    Transcript:
    --- START ---
    {text}
    --- END ---
      • rating is 0-5
      • category **must** be exactly one of the values shown above
    """

def get_default_report():
    keys = ["feedback", "keyTopics", "emotions",
            "sentiment", "output", "riskWords", "summary", "rating", "category"]
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

import subprocess, shlex, wave, contextlib
from mutagen import File as MutagenFile
from mutagen.id3 import ID3
from mutagen.mp4 import MP4
from typing import Optional, Tuple
from datetime import datetime
import logging

log = logging.getLogger(__name__)

def extract_audio_metadata(path: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Returns (recording_date_dd/mm/yyyy | None, duration_seconds | None).
    Falls back gracefully when tags or formats are missing.
    """
    rec_date: Optional[str] = None
    duration: Optional[float] = None

    # ------------------------------------------------------------------ #
    # 1) first attempt – mutagen (fast, no subprocess)
    # ------------------------------------------------------------------ #
    try:
        audio = MutagenFile(path, easy=False)
        if audio:
            # ---- duration -------------------------------------------------
            duration = getattr(audio.info, "length", None)

            # ---- date recorded  ------------------------------------------
            # MP3 / ID3
            if isinstance(audio, ID3) or hasattr(audio, "tags"):
                id3 = audio.tags or {}
                if "TDRC" in id3:
                    try:
                        d = id3["TDRC"].text[0]          # mutagen ID3 frame
                        if hasattr(d, "year"):
                            rec_date = datetime(d.year, d.month, d.day).strftime("%d/%m/%Y")
                        else:                            # string fallback
                            rec_date = str(d)[:10]
                    except Exception:
                        pass
                # MP4 / M4A
                if isinstance(audio, MP4) and "\xa9day" in audio.tags:
                    rec_date = str(audio.tags["\xa9day"][0])[:10].replace("-", "/")

            if rec_date:
                rec_date = normalise_date(rec_date)      # helper below
    except Exception as e:
        log.debug(f"mutagen failed on {path}: {e}")

    # ------------------------------------------------------------------ #
    # 2) second attempt – raw WAV header (uncompressed PCM only)
    # ------------------------------------------------------------------ #
    if duration is None and path.lower().endswith(".wav"):
        try:
            with contextlib.closing(wave.open(path)) as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate:
                    duration = frames / float(rate)
        except wave.Error as e:
            log.debug(f"wave fallback failed on {path}: {e}")

    # ------------------------------------------------------------------ #
    # 3) final attempt – ffprobe (works on *anything* if ffmpeg installed)
    # ------------------------------------------------------------------ #
    if duration is None or rec_date is None:
        try:
            cmd = (
                "ffprobe -v error -show_entries "
                "format=duration:format_tags=creation_time "
                "-of default=nw=1:nk=1 " + shlex.quote(path)
            )
            out = subprocess.check_output(cmd, shell=True, text=True).splitlines()
            if len(out) >= 1 and duration is None:
                duration = float(out[0])
            if len(out) >= 2 and rec_date is None:
                rec_date = normalise_date(out[1])
        except (subprocess.CalledProcessError, ValueError) as e:
            log.debug(f"ffprobe fallback failed on {path}: {e}")

    return rec_date, duration


# ------------------------------ helpers ------------------------------------ #
def normalise_date(raw: str) -> Optional[str]:
    """
    Accepts an arbitrary date string (YYYY‑MM‑DD, YYYY:MM:DD, etc.)
    and returns dd/mm/yyyy — or None if parse fails.
    """
    raw = raw.strip()[:10].replace(":", "-").replace("/", "-")
    try:
        d = datetime.fromisoformat(raw)
        return d.strftime("%d/%m/%Y")
    except ValueError:
        return None


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

    # --- post-processing ------------------------------------------------
    # 1) make sure riskWords is a non-empty list
    report["riskWords"] = ensure_list(report.get("riskWords"))
    if not report["riskWords"]:
        # fall-back: simple keyword sweep
        fallback_risks = ["refund", "cancel", "angry", "frustrated",
                          "lawsuit", "escalate", "complaint"]
        hits = [w for w in fallback_risks if w in full_text.lower()]
        report["riskWords"] = hits or ["none-found"]

    # 2) add / overwrite the category
    report["category"] = classify_call(full_text)





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
            date=rec_date,          # ★ persists to JSON on disk
            duration=duration_str,           #   (keep both names & types
             
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
        "type": report.get("category", "Sin categoría"),
        "duration": duration_str,
        "rating": report.get("rating", 0),
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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
    context = "\n".join([
        "You are a helpful assistant analysing a recorded call.",
        "--- Report ---",
        f"Feedback: {rpt.get('feedback','')}",
        f"KeyTopics: {', '.join(rpt.get('keyTopics', []))}",
        f"Emotions: {', '.join(rpt.get('emotions', []))}",
        f"Sentiment: {rpt.get('sentiment','')}",
        f"Output: {rpt.get('output','')}",
        # f"RiskWords: {rpt.get('riskWords','')}",
        f"RiskWords: {', '.join(ensure_list(rpt.get('riskWords')))}",
        f"Summary: {rpt.get('summary','')}",
        f"OCI sentiment: {oci_emotion}",
        "--- Transcript ---",
        transcript,
        # add logging to make sure risk words work properly and dont return empty string
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





















