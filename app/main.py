import sys
import os
import json
import uuid
import uvicorn
import logging
from datetime import date
import torch # Needed to check CUDA availability

# --- Import Google Gemini ---
import google.generativeai as genai

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

import oci
from oci.config import from_file
from oci.ai_language import AIServiceLanguageClient
from oci.ai_language.models import (
    BatchDetectLanguageSentimentsDetails,
    TextDocument
)

# Import our temp storage utility
from temp_storage import save_transcript, load_transcript # Assuming this file exists

# --- Import the new Transcription Module ---
from transcription_module import AudioTranscriber

# Basic Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Get logger instances for main app and the module if needed
logger = logging.getLogger(__name__)
# Ensure module logger propagates if handlers aren't set there explicitly
# logging.getLogger('transcription_module').propagate = True


app = FastAPI()

# CORS Setup
origins = ["http://localhost:3000"]  # Adjust or add your front-end URL as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = 'gemini-1.5-flash'
gemini_model = None
try:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.warning("Gemini API Key not configured. LLM features will be disabled.")
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        logger.info(f"Gemini AI SDK configured successfully with model {GEMINI_MODEL_NAME}.")
except Exception as e:
    logger.error(f"Failed to configure Gemini AI SDK: {e}", exc_info=True)
    gemini_model = None

# --- OCI Setup ---
oci_config = None
language_client = None
try:
    # Check if running inside OCI Function or Container Instance for instance principal
    if os.getenv('OCI_RESOURCE_PRINCIPAL_VERSION'):
        logger.info("Attempting to use OCI Resource Principal for authentication.")
        signer = oci.auth.signers.get_resource_principals_signer()
        language_client = AIServiceLanguageClient(config={}, signer=signer)
    else:
        logger.info("Attempting to use OCI config file for authentication.")
        oci_config = from_file()
        language_client = AIServiceLanguageClient(oci_config)
    logger.info("OCI SDK configured successfully.")
except Exception as e:
    logger.error(f"Failed to configure OCI SDK: {e}", exc_info=True)
    language_client = None

# --- Transcription Module Setup ---
# Determine device and compute type for WhisperX
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("##############################################################")
    logger.warning("## WARNING: HF_TOKEN environment variable not set.          ##")
    logger.warning("## Speaker diarization requires a Hugging Face token.       ##")
    logger.warning("## Diarization features will likely fail.                   ##")
    logger.warning("## Set HF_TOKEN in your environment.                        ##")
    logger.warning("##############################################################")
    # Decide if you want to stop the server or continue without diarization
    # sys.exit("ERROR: HF_TOKEN is required. Server stopping.") # Option to enforce

# Initialize the transcriber globally to load models only once
transcriber = None
try:
    # Dynamically set device based on availability
    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float16 for GPU, int8 for CPU generally
    whisper_compute_type = "float16" if whisper_device == "cuda" else "int8"
    # You might want to make model size configurable via env var too
    whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "base") # Default to "base"

    logger.info(f"Attempting to initialize AudioTranscriber globally (Model: {whisper_model_size}, Device: {whisper_device}, Compute: {whisper_compute_type})...")
    transcriber = AudioTranscriber(
        model_name=whisper_model_size,
        device=whisper_device,
        compute_type=whisper_compute_type
        # hf_token is read from env within the class
    )
    logger.info("AudioTranscriber initialized successfully.")
except Exception as e:
    logger.error(f"##############################################################")
    logger.error(f"## FATAL: Failed to initialize AudioTranscriber globally.   ##")
    logger.error(f"## Error: {e}                                               ##")
    logger.error(f"## Transcription endpoint will not work. Check logs above.  ##")
    logger.error(f"##############################################################", exc_info=True)
    # Depending on severity, you might want to exit or let it run with transcription disabled
    # sys.exit("ERROR: Failed to load transcription model. Server stopping.")


@app.get("/")
def read_root():
    return {"message": "Audio Analysis API Ready"}




# ----------- Prompt creation (include riskWords) -----------
# (Keep create_llm_prompt, get_default_report, generate_call_report_with_llm as they were)
# ... (rest of the LLM prompt and report generation functions remain the same)
def create_llm_prompt(transcript_text: str) -> str:
    """
    Creates the detailed prompt for the LLM,
    now including a 'riskWords' field in the JSON.
    """
    prompt = f"""
        You are an expert call analyst. Based on the provided call transcript (which includes speaker labels like SPEAKER_00:, SPEAKER_01:), generate a structured report containing the following sections:
        - Feedback
        - Key Topics
        - Emotions
        - Sentiment
        - Output (Resolution)
        - Risk Words (any escalation threats, negative expressions, cancellations, or other 'risky' words the customer used)
        - Summary

        Analyze the transcript below:
        --- START TRANSCRIPT ---
        {transcript_text}
        --- END TRANSCRIPT ---

        Generate your analysis strictly in the following JSON format. Do not include any text before or after the JSON object:

        {{
        "feedback": "Specific feedback from or about the call.",
        "keyTopics": [
            "List of main topics in the conversation."
        ],
        "emotions": [
            "List of the primary emotions the speaker(s) showed."
        ],
        "sentiment": "Overall sentiment of the call. E.g., Negative, Positive, Neutral, Mixed, with short reasoning if necessary.",
        "output": "Next steps or resolution details from the conversation.",
        "riskWords": "Highlight if any language is escalatory or signals risk (if none, say 'None')",
        "summary": "One or two-sentence summary of the entire call."
        }}
        """
    return prompt

def get_default_report():
    return {
        "feedback": "Error generating feedback.",
        "keyTopics": ["Error generating key topics."],
        "emotions": ["Error generating emotions."],
        "sentiment": "Error generating sentiment.",
        "output": "Error generating output.",
        "riskWords": "Error generating risk words.",
        "summary": "Error generating summary."
    }

def generate_call_report_with_llm(transcript_text: str) -> Dict[str, Any]:
    """
    Send the transcript to the Gemini LLM to get a structured report (JSON).
    """
    default_report = get_default_report()

    if not gemini_model:
        logger.error("Gemini model not available for generating report.")
        return default_report

    prompt = create_llm_prompt(transcript_text)

    try:
        logger.info("Sending request to Gemini LLM for structured report.")
        response = gemini_model.generate_content(prompt)
        llm_response_content = response.text

        if not llm_response_content:
            logger.error("Gemini LLM response content is empty.")
            return default_report

        cleaned_content = llm_response_content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3]

        report_data = json.loads(cleaned_content.strip())

        expected_keys = [
            "feedback", "keyTopics", "emotions", "sentiment", "output", "riskWords", "summary"
        ]
        for key in expected_keys:
            if key not in report_data:
                report_data[key] = report_data.get(key, f"Missing '{key}' in LLM response.")
                logger.warning(f"Key '{key}' missing in LLM JSON response. Using default.")

        return report_data

    except json.JSONDecodeError as json_err:
        logger.error(f"Failed to decode JSON response from LLM: {json_err}")
        logger.error(f"LLM raw response was: {llm_response_content}")
        return default_report
    except Exception as e:
        logger.error(f"Error generating call report with Gemini LLM: {e}", exc_info=True)
        return default_report


# ----------- Upload endpoint (uses transcription_module) -----------
@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Handles audio upload, performs transcription with speaker diarization,
    runs analysis (OCI, LLM), and returns a structured result.
    """
    if transcriber is None:
        logger.error("AudioTranscriber is not initialized. Cannot process upload.")
        raise HTTPException(status_code=503, detail="Transcription service is not available.")

    # 1) Save file temporarily
    # Using a simpler temp filename structure
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True) # Ensure dir exists
    # Create a unique filename within the temp dir
    temp_filename = os.path.join(temp_dir, f"audio_{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")


    try:
        with open(temp_filename, "wb") as f:
            content = await file.read()
            if not content:
                 raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            f.write(content)
        logger.info(f"Saved uploaded file to {temp_filename}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}", exc_info=True)
        # Clean up if save failed partially
        if os.path.exists(temp_filename):
            try: os.remove(temp_filename)
            except OSError: pass
        raise HTTPException(status_code=500, detail=f"Failed to save audio file: {e}")

    # 2) Transcribe & Diarize using the module
    diarized_transcript: List[Dict[str, str]] = []
    try:
        # Call the method from the globally initialized transcriber instance
        # You can add min/max speakers here if known: transcriber.transcribe_and_diarize(temp_filename, min_speakers=2)
        diarized_transcript = transcriber.transcribe_and_diarize(temp_filename)

        # Check if transcription/diarization failed (indicated by specific content perhaps)
        if not diarized_transcript or "[Error" in diarized_transcript[0].get("text", "") or "[failed" in diarized_transcript[0].get("text", ""):
             logger.error(f"Transcription/diarization failed for {temp_filename}. Result: {diarized_transcript}")
             # Decide how to proceed: raise error or continue with limited data
             # Raising error is cleaner if transcript is essential
             raise HTTPException(status_code=500, detail=f"Transcription/Diarization failed: {diarized_transcript[0]['text'] if diarized_transcript else 'Unknown error'}")

        logger.info(f"Transcription and diarization complete for {temp_filename}.")

    except Exception as e:
        logger.error(f"Error calling transcription module: {e}", exc_info=True)
        # Clean up temp file on error
        if os.path.exists(temp_filename):
            try: os.remove(temp_filename)
            except OSError as rm_err: logger.warning(f"Could not remove temp file {temp_filename} after error: {rm_err}")
        raise HTTPException(status_code=500, detail=f"Transcription process failed: {e}")

    # Get the full text string with speaker labels for analysis steps
    # Use the helper method from the transcriber instance
    full_transcript_text = transcriber.get_full_text(diarized_transcript)

    # 3) OCI Sentiment Analysis (optional) - using the full_transcript_text
    oci_emotion_text = "N/A"
    oci_aspects_list = []
    if language_client and full_transcript_text and "Error" not in full_transcript_text: # Check OCI client and valid transcript
        try:
            documents = [TextDocument(key="doc1", text=full_transcript_text, language_code="en")]
            sentiment_request = BatchDetectLanguageSentimentsDetails(documents=documents)
            logger.info("Sending request to OCI Language for sentiment analysis...")
            response = language_client.batch_detect_language_sentiments(
                batch_detect_language_sentiments_details=sentiment_request
            )
            if response.data.documents and len(response.data.documents) > 0:
                doc_result = response.data.documents[0]
                oci_emotion_text = getattr(doc_result, 'document_sentiment', None)
                if oci_emotion_text:
                    oci_emotion_text = oci_emotion_text.label
                elif doc_result.aspects:
                     oci_emotion_text = doc_result.aspects[0].sentiment if doc_result.aspects[0].sentiment else "Mixed (aspects found)"
                else:
                    oci_emotion_text = "Neutral (default)"

                if doc_result.aspects:
                    oci_aspects_list = [{"text": a.text, "sentiment": a.sentiment, "scores": a.scores} for a in doc_result.aspects]
                logger.info(f"OCI Sentiment analysis complete. Overall: {oci_emotion_text}")
            else:
                 logger.warning("OCI Sentiment analysis returned no document results.")
                 oci_emotion_text = "N/A (OCI empty response)"

        except oci.exceptions.ServiceError as oci_err:
            logger.error(f"OCI sentiment analysis failed (Service Error): {oci_err.status} - {oci_err.message}", exc_info=True)
            oci_emotion_text = f"Error (OCI {oci_err.status})"
        except Exception as e:
            logger.error(f"OCI sentiment analysis failed (General Error): {e}", exc_info=True)
            oci_emotion_text = "Error (OCI failure)"
    elif not language_client:
        logger.warning("OCI Language client not configured. Skipping OCI sentiment analysis.")
    elif not full_transcript_text or "Error" in full_transcript_text:
         logger.warning("Skipping OCI sentiment analysis due to invalid/failed transcript.")
         oci_emotion_text = "N/A (Invalid Transcript)"


    # 4) Generate the structured LLM report using Gemini - using the full_transcript_text
    report_data = generate_call_report_with_llm(full_transcript_text)

    # 5) Save in temp storage (optional)
    transcript_id = str(uuid.uuid4())
    try:
        # --- CORRECTED CALL ---
        # Ensure the variables `diarized_transcript` and `full_transcript_text`
        # exist and hold the correct data before this point.
        save_transcript(
            transcript_id=transcript_id,
            diarized_transcript=diarized_transcript, # Use the structured list
            full_transcript_text=full_transcript_text, # Use the full text string
            report_data=report_data,
            oci_emotion=oci_emotion_text,
            oci_aspects=oci_aspects_list
        )
        # --- END OF CORRECTION ---
        logger.info(f"Transcript data saved for ID: {transcript_id}")
    # Keep the exception handling as it is
    except NameError:
         logger.warning("`save_transcript` function not found in `temp_storage`. Skipping save.")
    except FileNotFoundError as e: # Added specific exception from temp_storage
        logger.error(f"Error saving transcript {transcript_id}: {e}", exc_info=True)
    except (IOError, TypeError, ValueError) as e: # Added specific exceptions from temp_storage
        logger.error(f"Error saving transcript {transcript_id}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to save transcript data for {transcript_id}: {e}", exc_info=True)


    # Remove the temp audio file *after* all processing
    if os.path.exists(temp_filename):
        try:
            os.remove(temp_filename)
            logger.info(f"Removed temp audio file: {temp_filename}")
        except OSError as e:
            logger.warning(f"Could not remove temp file {temp_filename}: {e}")

    # 6) Build final JSON response
    current_date_str = date.today().strftime("%d/%m/%Y")

    # TODO: Calculate duration - use a library like librosa or ffprobe on temp_filename *before* deleting
    # Example using ffprobe (requires ffprobe installed):
    duration_str = "N/A"
    # try:
    #     import subprocess
    #     cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{temp_filename}\""
    #     duration_seconds = float(subprocess.check_output(cmd, shell=True))
    #     duration_str = time.strftime('%H:%M:%S', time.gmtime(duration_seconds)) # Format as HH:MM:SS
    #     logger.info(f"Calculated audio duration: {duration_str}")
    # except Exception as dur_e:
    #     logger.warning(f"Could not calculate audio duration: {dur_e}")


    final_response = {
        "id": transcript_id,
        "name": file.filename if file else "Unknown Filename",
        "date": current_date_str,
        "type": "Sin categoría",
        "duration": duration_str, # Placeholder, implement calculation if needed
        "rating": 0,
        "report": report_data, # Contains all keys from LLM or defaults
        "transcript": diarized_transcript # Use the structured list directly
    }

    # Return final JSON
    return final_response


# ---------- Chat endpoint ----------
class ChatRequest(BaseModel):
    transcript_id: str
    messages: List[Dict[str, str]] # Expecting {"role": "user/assistant", "content": "..."}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    Allows follow-up Q&A referencing the saved transcript+report using Gemini.
    """
    if not gemini_model:
        logger.error("Gemini model not available for chat.")
        raise HTTPException(status_code=503, detail="LLM service (Gemini) not configured or unavailable.")

    transcript_obj = None
    try:
        transcript_obj = load_transcript(request.transcript_id)
    except NameError:
        logger.error("`load_transcript` function not found in `temp_storage`.")
        raise HTTPException(status_code=500, detail="Server configuration error: temp storage unavailable.")
    except FileNotFoundError:
        logger.warning(f"Transcript file not found for id: {request.transcript_id}")
        raise HTTPException(status_code=404, detail="Transcript not found")
    except Exception as e:
        logger.error(f"Error loading transcript {request.transcript_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load transcript data.")


    if not transcript_obj: # Should be caught by FileNotFoundError, but double-check
        logger.warning(f"Transcript data is empty for id: {request.transcript_id}")
        raise HTTPException(status_code=404, detail="Transcript not found or is empty")

    # --- Retrieve necessary data from the loaded object ---
    # Prefer the full text with speaker labels if saved, otherwise reconstruct it
    # Assuming temp_storage saves 'full_transcript_text' now
    transcript_text = transcript_obj.get("full_transcript_text")
    if not transcript_text:
        # Fallback: Reconstruct from diarized list if full text wasn't saved
        diarized_list = transcript_obj.get("diarized_transcript", [])
        if diarized_list:
            transcript_text = "\n".join([f"{entry['speaker']}: {entry['text']}" for entry in diarized_list])
            logger.warning(f"Reconstructed full transcript text for chat context for ID: {request.transcript_id}")
        else:
             transcript_text = "[Transcript text unavailable]"
             logger.error(f"Could not find or reconstruct transcript text for chat context for ID: {request.transcript_id}")


    report = transcript_obj.get("report_data", {}) # Use the key saved by save_transcript
    oci_emotion = transcript_obj.get("oci_emotion", "N/A")

    # --- Build context for Gemini ---
    # Provide the report and the transcript text (with speaker labels)
    context_parts = [
        "You are a helpful assistant analyzing a specific recorded call.",
        "Here is the context for the call you need to answer questions about:",
        f"\n--- Call Analysis Report ---",
        f"Feedback: {report.get('feedback', 'N/A')}",
        f"Key Topics: {'; '.join(report.get('keyTopics', ['N/A']))}",
        f"Detected Emotions: {'; '.join(report.get('emotions', ['N/A']))}",
        f"Overall Sentiment: {report.get('sentiment', 'N/A')}",
        f"Call Output/Resolution: {report.get('output', 'N/A')}",
        f"Risk Words: {report.get('riskWords', 'N/A')}",
        f"Summary: {report.get('summary', 'N/A')}",
        f"Additional OCI Sentiment: {oci_emotion}",
        # Include the full transcript text with speaker labels for better context
        f"\n--- Full Call Transcript ---",
        transcript_text if transcript_text else "[Transcript not available]",
        # Instruction
        "\nBased *only* on the provided transcript and the analysis report, answer the user's questions concisely. Do not invent information not present in the provided context."
    ]
    initial_context = "\n".join(context_parts)

    # --- Format message history for Gemini ---
    gemini_messages = []
    gemini_messages.append({'role': 'user', 'parts': [initial_context]})
    gemini_messages.append({'role': 'model', 'parts': ["Okay, I have reviewed the call context including the full transcript with speaker labels. How can I help you?"]})

    for msg in request.messages:
        role = "model" if msg.get("role") == "assistant" else "user"
        content = msg.get("content", "")
        if content:
             gemini_messages.append({'role': role, 'parts': [content]})

    if not gemini_messages or gemini_messages[-1]['role'] != 'user':
         logger.warning("Chat request did not end with a user message. Appending a default prompt.")
         gemini_messages.append({'role': 'user', 'parts': ["Please provide information based on the context."]})

    # --- Call Gemini API ---
    try:
        logger.info(f"Sending chat request to Gemini LLM for transcript {request.transcript_id}")
        response = gemini_model.generate_content(gemini_messages)
        assistant_reply = response.text

        if not assistant_reply:
            logger.warning("Gemini LLM returned empty reply for chat request.")
            try: logger.info(f"Gemini Prompt Feedback: {response.prompt_feedback}")
            except Exception: pass
            return {"assistant_message": "Sorry, I couldn't generate a response for that request."}

        return {"assistant_message": assistant_reply}

    except Exception as e:
        logger.error(f"Exception calling Gemini LLM during chat for transcript {request.transcript_id}: {e}", exc_info=True)
        error_detail = f"LLM service error: {str(e)}"
        if "API key not valid" in str(e):
            error_detail = "LLM service error: Invalid API Key."
        raise HTTPException(status_code=503, detail=error_detail)


# ---------- Test OCI endpoint ----------
# (Keep /test_oci as it was)
@app.get("/test_oci")
def test_oci_call():
    """
    Quick test route for checking OCI doc-level sentiment.
    """
    if not language_client:
        logger.error("OCI language client not configured.")
        return {"error": "OCI client not configured."}

    text = "I am very unhappy with the service, it was slow and the product broke quickly. However, the support agent Alex was trying to be helpful."
    documents = [TextDocument(key="doc1", text=text, language_code="en")]
    sentiment_request = BatchDetectLanguageSentimentsDetails(documents=documents)

    try:
        logger.info("Sending test request to OCI Language Service...")
        response = language_client.batch_detect_language_sentiments(
            batch_detect_language_sentiments_details=sentiment_request
        )
        logger.info(f"/test_oci OCI response status: {response.status}")
        try:
            return response.data # FastAPI might handle OCI models
        except Exception:
            return {"data": str(response.data)} # Basic conversion fallback
    except oci.exceptions.ServiceError as oci_err:
         logger.error(f"/test_oci OCI Service Error: Status {oci_err.status}, Code {oci_err.code}, Message {oci_err.message}", exc_info=True)
         return {"error": f"OCI Service Error: {oci_err.message} (Status: {oci_err.status})"}
    except Exception as ex:
        logger.error(f"/test_oci generic exception: {ex}", exc_info=True)
        return {"error": str(ex)}



import unittest

class TestOciCall():
    print("Testing OCI call")

    def test_oci_no_lang(self):
        print("Test of oci with no lang...")

    def test_oci_no_lang(self):
        print("Test of oci with no lang...")


# ---------- Run server if executed directly ----------
if __name__ == "__main__":
    # Pre-run checks
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
         print("--- WARNING: GEMINI_API_KEY is not set. LLM features will not work. ---")
    if gemini_model is None and (GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE"):
         print("--- WARNING: Gemini model failed to initialize despite API key being present. Check configuration and logs. ---")
    if transcriber is None:
        print("--- ERROR: AudioTranscriber failed to initialize. Transcription endpoint WILL NOT WORK. Check logs. ---")
        # Optionally exit if transcription is critical
        # sys.exit(1)
    if not HF_TOKEN:
        print("--- WARNING: HF_TOKEN is not set. Speaker diarization WILL LIKELY FAIL. ---")

    logger.info("Starting FastAPI server on host 0.0.0.0, port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") # Set uvicorn log level too

