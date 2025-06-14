"""
main.py 

"""

import os
import sys
import json
import uuid
import gc
import logging
from enum import Enum
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, datetime          
from mutagen import File as MutagenFile

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
import mimetypes

# Import our new AI provider system
from ai_providers import AIProviderFactory, AIProvider, test_provider
from db import query
from temp_storage import save_transcript, load_transcript  
from rag_chat import rag_chat as rag_chat_function, process_rag_chat_request, create_rag_chat_request 
from request_tracker import request_tracker  

import subprocess, shlex, wave, contextlib
from mutagen.id3 import ID3
from mutagen.mp4 import MP4

# Keep the existing secure prompts for now
from secure_prompts import SecurePromptManager

# --------------------------------------------------------------------------- #
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------------------------------------------------------------- #
# Pydantic Models (IMPORTANT: Define these before use)
class ChatRequest(BaseModel):
    transcript_id: str
    messages: List[Dict[str, str]]

class RAGChatRequest(BaseModel):
    question: str
    call_ids: List[str]

class RAGChatStartRequest(BaseModel):
    question: str
    call_ids: List[str]

class RAGChatStartResponse(BaseModel):
    request_id: str
    status: str

class RAGChatStatusResponse(BaseModel):
    request_id: str
    status: str
    created_at: float
    updated_at: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# --------------------------------------------------------------------------- #
# Updated Runtime Settings with AI Provider Support
class TranscriptionEngine(str, Enum):
    whisperx = "whisperx"
    ai_provider = "ai_provider"  # Use configured AI provider

class RuntimeSettings(BaseModel):
    # Existing settings
    transcription_engine: TranscriptionEngine = TranscriptionEngine.ai_provider
    llm_model: str = "gemini-1.5-flash"
    
    # New AI Provider settings
    ai_provider: str = "google"  # google, openai, claude, ollama, lm_studio, custom
    api_key: str = ""
    base_url: Optional[str] = None  # For custom endpoints, Ollama, LM Studio
    
    # Additional settings
    temperature: float = 0.1
    max_tokens: int = 4000
    
    @validator('ai_provider')
    def validate_provider(cls, v):
        valid_providers = [provider.value for provider in AIProvider]
        if v not in valid_providers:
            raise ValueError(f"Invalid AI provider. Must be one of: {valid_providers}")
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

# Initialize with default settings
runtime_settings = RuntimeSettings(
    ai_provider="google",
    api_key=os.getenv("GEMINI_API_KEY", ""),
    llm_model="gemini-1.5-flash"
)

# Global AI provider instance
current_ai_provider = None

def get_ai_provider():
    """Get or create the current AI provider instance"""
    global current_ai_provider
    
    if (current_ai_provider is None or 
        current_ai_provider.model_name != runtime_settings.llm_model or
        current_ai_provider.api_key != runtime_settings.api_key):
        
        try:
            provider_enum = AIProvider(runtime_settings.ai_provider)
            current_ai_provider = AIProviderFactory.create_provider(
                provider_enum,
                runtime_settings.api_key,
                runtime_settings.llm_model,
                runtime_settings.base_url
            )
            logger.info(f"AI Provider initialized: {runtime_settings.ai_provider} with model {runtime_settings.llm_model}")
        except Exception as e:
            logger.error(f"Failed to initialize AI provider: {e}")
            raise HTTPException(status_code=500, detail=f"AI Provider initialization failed: {e}")
    
    return current_ai_provider

# --------------------------------------------------------------------------- #
# Settings endpoints
@app.post("/settings", response_model=RuntimeSettings)
async def update_runtime_settings(payload: RuntimeSettings):
    global current_ai_provider
    
    # Update global settings
    runtime_settings.transcription_engine = payload.transcription_engine
    runtime_settings.llm_model = payload.llm_model
    runtime_settings.ai_provider = payload.ai_provider
    runtime_settings.api_key = payload.api_key
    runtime_settings.base_url = payload.base_url
    runtime_settings.temperature = payload.temperature
    runtime_settings.max_tokens = payload.max_tokens
    
    # Reset AI provider to force recreation with new settings
    current_ai_provider = None
    
    logger.info(f"Settings updated → Provider: {runtime_settings.ai_provider}, Model: {runtime_settings.llm_model}")
    return runtime_settings

@app.get("/settings", response_model=RuntimeSettings)
def read_runtime_settings():
    return runtime_settings

@app.get("/settings_ui")
def settings_ui():
    """Serve the AI Provider settings UI"""
    return FileResponse("static/settings.html")

@app.get("/providers")
def get_available_providers():
    """Get list of available AI providers"""
    return {
        "providers": AIProviderFactory.get_available_providers(),
        "current": runtime_settings.ai_provider
    }

@app.post("/test_provider")
async def test_ai_provider(payload: Dict[str, Any]):
    """Test AI provider configuration"""
    try:
        result = await test_provider(
            provider_type=payload.get("provider", runtime_settings.ai_provider),
            api_key=payload.get("api_key", runtime_settings.api_key),
            model_name=payload.get("model_name", runtime_settings.llm_model),
            base_url=payload.get("base_url", runtime_settings.base_url)
        )
        return result
    except Exception as e:
        logger.error(f"Provider test failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# --------------------------------------------------------------------------- #
# Helper functions using AI Provider
CATEGORIES = ["Sales", "Customer Support",
               "Other"]

def ensure_list(val):
    """Accept str | list | None → list[str]."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        return [x.strip() for x in val.split(",") if x.strip()]
    return []

async def classify_call(text: str) -> str:
    """Quick classification using the configured AI provider."""
    try:
        provider = get_ai_provider()
        prompt = (
            "Choose the single most relevant category for this call from "
            f"{CATEGORIES}. Just return the word.\n\nTranscript:\n{text}"
        )
        response = await provider.generate_text(prompt, temperature=0.1)
        guess = response.strip().splitlines()[0] if response else "Other"
        return guess if guess in CATEGORIES else "Other"
    except Exception as e:
        logger.warning(f"Category classification failed: {e}")
        return "Other"

# --------------------------------------------------------------------------- #
# Keep existing metadata extraction functions
def extract_audio_metadata(path: str) -> Tuple[Optional[str], Optional[float]]:
    """Returns (recording_date_dd/mm/yyyy | None, duration_seconds | None)."""
    rec_date: Optional[str] = None
    duration: Optional[float] = None

    try:
        audio = MutagenFile(path, easy=False)
        if audio:
            duration = getattr(audio.info, "length", None)
            
            # Extract date from various formats
            if isinstance(audio, ID3) or hasattr(audio, "tags"):
                id3 = audio.tags or {}
                if "TDRC" in id3:
                    try:
                        d = id3["TDRC"].text[0]
                        if hasattr(d, "year"):
                            rec_date = datetime(d.year, d.month, d.day).strftime("%d/%m/%Y")
                        else:
                            rec_date = str(d)[:10]
                    except Exception:
                        pass
                        
                if isinstance(audio, MP4) and "\xa9day" in audio.tags:
                    rec_date = str(audio.tags["\xa9day"][0])[:10].replace("-", "/")

            if rec_date:
                rec_date = normalise_date(rec_date)
    except Exception as e:
        logger.debug(f"mutagen failed on {path}: {e}")

    # WAV fallback
    if duration is None and path.lower().endswith(".wav"):
        try:
            with contextlib.closing(wave.open(path)) as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate:
                    duration = frames / float(rate)
        except wave.Error as e:
            logger.debug(f"wave fallback failed on {path}: {e}")

    # ffprobe fallback
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
            logger.debug(f"ffprobe fallback failed on {path}: {e}")

    return rec_date, duration

def normalise_date(raw: str) -> Optional[str]:
    """Accepts date string and returns dd/mm/yyyy format."""
    raw = raw.strip()[:10].replace(":", "-").replace("/", "-")
    try:
        d = datetime.fromisoformat(raw)
        return d.strftime("%d/%m/%Y")
    except ValueError:
        return None

# Update the SecurePromptManager integration
@lru_cache(maxsize=1)
def get_secure_prompt_manager() -> 'SecurePromptManager':
    """Get SecurePromptManager instance with current AI provider settings"""
    from secure_prompts import SecurePromptManager
    return SecurePromptManager(
        provider_type=runtime_settings.ai_provider,
        api_key=runtime_settings.api_key,
        model_name=runtime_settings.llm_model,
        base_url=runtime_settings.base_url
    )

# --------------------------------------------------------------------------- #
# Updated upload endpoint
@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file received.")

    # Clear cache to ensure fresh SecurePromptManager with current settings
    get_secure_prompt_manager.cache_clear()

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
        logger.error(f"File save failed: {e}")
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    # Extract metadata
    rec_date, duration_sec = extract_audio_metadata(temp_path)
    if not rec_date:
        rec_date = datetime.fromtimestamp(os.path.getmtime(temp_path)).strftime("%d/%m/%Y")

    def fmt_dur(s: Optional[float]) -> str:
        if s is None:
            return "N/A"
        h, m = divmod(int(s), 3600)
        m, s = divmod(m, 60)
        return f"{h:02}:{m:02}:{s:02}"

    duration_str = fmt_dur(duration_sec)

    # Transcription using SecurePromptManager
    try:
        if runtime_settings.transcription_engine == TranscriptionEngine.ai_provider:
            spm = get_secure_prompt_manager()
            mime, _ = mimetypes.guess_type(temp_path)
            with open(temp_path, "rb") as f:
                audio_data = f.read()
            diarized = await spm.secure_transcribe_audio(audio_data, mime or "application/octet-stream")
        else:
            # Fallback to WhisperX if needed
            raise HTTPException(status_code=501, detail="WhisperX not implemented in this version")
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    # Create full text and process with SecurePromptManager
    full_text = "\n".join(f"{d['speaker']}: {d['text']}" for d in diarized)

    # Use SecurePromptManager for speaker identification and report generation
    spm = get_secure_prompt_manager()
    mappings = await spm.secure_identify_speakers(full_text)
    speaker_map = {m["old"]: m["new"] for m in mappings}

    for seg in diarized:
        if seg["speaker"] in speaker_map:
            seg["speaker"] = speaker_map[seg["speaker"]]

    full_text = "\n".join(f"{d['speaker']}: {d['text']}" for d in diarized)

    # Generate report using SecurePromptManager
    report = await spm.secure_generate_report(full_text)

    # Post-processing
    report["riskWords"] = ensure_list(report.get("riskWords"))
    if not report["riskWords"]:
        fallback_risks = ["refund", "cancel", "angry", "frustrated",
                          "lawsuit", "escalate", "complaint"]
        hits = [w for w in fallback_risks if w in full_text.lower()]
        report["riskWords"] = hits or ["none-found"]

    # Add category classification
    report["category"] = await classify_call(full_text)

    # Save transcript
    transcript_id = str(uuid.uuid4())
    try:
        save_transcript(
            transcript_id=transcript_id,
            diarized_transcript=diarized,
            full_transcript_text=full_text,
            report_data=report,
            oci_emotion="N/A",
            oci_aspects=[],
            date=rec_date,
            duration=duration_str,
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
# Updated chat endpoint
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
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
        f"RiskWords: {', '.join(ensure_list(rpt.get('riskWords')))}",
        f"Summary: {rpt.get('summary','')}",
        f"OCI sentiment: {oci_emotion}",
        "--- Transcript ---",
        transcript,
    ])

    # Convert messages for AI provider
    chat_messages = [
        {"role": "system", "content": context},
        {"role": "assistant", "content": "OK – how can I assist?"},
    ] + [
        {"role": msg["role"], "content": msg["content"]}
        for msg in req.messages
    ]

    try:
        provider = get_ai_provider()
        response = await provider.generate_chat(
            chat_messages, 
            temperature=runtime_settings.temperature
        )
        return {"assistant_message": response or "Sorry, I have no reply."}
    except Exception as e:
        logger.error(f"AI provider chat failed: {e}")
        raise HTTPException(status_code=503, detail="AI service error.")

# --------------------------------------------------------------------------- #
# RAG endpoints
@app.get("/rag")
def rag_chat_ui():
    """Serve the RAG chat UI HTML page"""
    return FileResponse("static/rag_chat.html")

@app.get("/rag_polling")
def rag_chat_polling_ui():
    """Serve the polling-based RAG chat UI HTML page"""
    return FileResponse("static/rag_chat_polling.html")

@app.post("/rag_chat")
async def rag_chat_endpoint(req: RAGChatRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not req.call_ids:
        raise HTTPException(status_code=400, detail="At least one call ID must be provided.")

    try:
        # Update RAG chat to use current AI provider settings
        response = await rag_chat_function(
            question=req.question,
            call_ids=req.call_ids,
            model_name=runtime_settings.llm_model,
            api_key=runtime_settings.api_key  # Use current API key
        )
        return response
    except Exception as e:
        logger.error(f"RAG chat failed: {e}")
        raise HTTPException(status_code=503, detail="RAG chat service error.")

@app.post("/rag_chat/start", response_model=RAGChatStartResponse)
async def start_rag_chat_endpoint(req: RAGChatStartRequest, background_tasks: BackgroundTasks):
    if not req.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not req.call_ids:
        raise HTTPException(status_code=400, detail="At least one call ID must be provided.")

    try:
        request_id = create_rag_chat_request()
        background_tasks.add_task(
            process_rag_chat_request,
            request_id=request_id,
            question=req.question,
            call_ids=req.call_ids,
            model_name=runtime_settings.llm_model,
            api_key=runtime_settings.api_key
        )
        status = request_tracker.get_request_status(request_id)
        return {
            "request_id": request_id,
            "status": status["status"]
        }
    except Exception as e:
        logger.error(f"Failed to start RAG chat request: {e}")
        raise HTTPException(status_code=503, detail="Failed to start RAG chat request.")

@app.get("/rag_chat/status/{request_id}", response_model=RAGChatStatusResponse)
async def get_rag_chat_status(request_id: str):
    status = request_tracker.get_request_status(request_id)
    if not status:
        raise HTTPException(status_code=404, detail="Request not found.")
    return status

# --------------------------------------------------------------------------- #
# Keep existing endpoints
@app.get("/")
def root():
    return {"message": "Audio Analysis API ready with AI Provider support."}

# Test endpoints
@app.get("/testdb")
async def get_users():
    sql = "SELECT * FROM CALLS;"
    return await query(sql)

@app.get("/health")
async def health_check():
    """Health check endpoint that tests AI provider connection"""
    try:
        provider = get_ai_provider()
        test_response = await provider.generate_text(
            "Respond with just 'OK' if you can hear me.",
            temperature=0.1
        )
        return {
            "status": "healthy",
            "provider": runtime_settings.ai_provider,
            "model": runtime_settings.llm_model,
            "supports_audio": provider.supports_audio(),
            "test_response": test_response[:50] if test_response else "No response"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "provider": runtime_settings.ai_provider,
            "model": runtime_settings.llm_model
        }

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 Starting Audio Analysis API with Multi-Provider AI Support")
    logger.info("=" * 60)
    logger.info(f"📡 Current AI Provider: {runtime_settings.ai_provider}")
    logger.info(f"🧠 Current Model: {runtime_settings.llm_model}")
    logger.info(f"🎙️  Transcription Engine: {runtime_settings.transcription_engine}")
    logger.info(f"🌡️  Temperature: {runtime_settings.temperature}")
    logger.info(f"📝 Max Tokens: {runtime_settings.max_tokens}")
    
    if runtime_settings.base_url:
        logger.info(f"🔗 Base URL: {runtime_settings.base_url}")
    
    logger.info("=" * 60)
    logger.info("🌐 Web Interface available at:")
    logger.info("   • Main API: http://localhost:8000")
    logger.info("   • Settings UI: http://localhost:8000/settings_ui")
    logger.info("   • Health Check: http://localhost:8000/health")
    logger.info("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")