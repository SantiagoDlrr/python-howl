import sys
import os
import json
import uuid
# import requests # will need for local integration, maybe lol
import whisper
import uvicorn
import logging
from datetime import date # Added for dynamic date

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
# IMPORTANT: Replace with your actual API key.
# SECURITY WARNING: Hardcoding keys is not recommended for production. Use environment variables.
GEMINI_API_KEY = ""

import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL_NAME = 'gemini-1.5-flash' # Or 'gemini-pro', etc.

gemini_model = None
try:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.warning("Gemini API Key not configured. LLM features will be disabled.")
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        # Optional: Test call to ensure configuration is valid (might incur cost)
        # gemini_model.generate_content("test")
        logger.info(f"Gemini AI SDK configured successfully with model {GEMINI_MODEL_NAME}.")
except Exception as e:
    logger.error(f"Failed to configure Gemini AI SDK: {e}")
    gemini_model = None # Mark as unusable


# OCI setup
try:
    oci_config = from_file()  # If you have an OCI config file
    language_client = AIServiceLanguageClient(oci_config)
    logger.info("OCI SDK configured successfully.")
except Exception as e:
    logger.error(f"Failed to configure OCI SDK: {e}")
    language_client = None  # None if setup fails


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI with Docker Compose"}


# ----------- Utility: parse transcript text into array of {speaker, text} -----------
def parse_transcript_to_list(full_text: str):
    """
    Naive parser that splits a conversation into a list of {speaker, text}.
    Assumes lines start with 'SpeakerName:' and continues until next speaker or text ends.
    """
    lines = full_text.strip().split("\n")
    transcript_entries = []

    current_speaker = None
    current_text = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check if line looks like "Something:"
        if ":" in line:
            # e.g., "Alex:", "Jamie:", etc.
            parts = line.split(":", 1)
            possible_speaker = parts[0].strip()
            rest = parts[1].strip() if len(parts) > 1 else ""

            # If the line indeed starts a new speaker:
            if current_speaker is not None and current_text:
                # Save the previous speaker chunk
                transcript_entries.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text).strip()
                })
            current_speaker = possible_speaker
            current_text = [rest] if rest else []
        else:
            # Continuation of current speaker's text
            current_text.append(line)

    # If there's leftover text at the end
    if current_speaker and current_text:
        transcript_entries.append({
            "speaker": current_speaker,
            "text": " ".join(current_text).strip()
        })

    return transcript_entries


# ----------- Prompt creation (include riskWords) -----------
def create_llm_prompt(transcript_text: str) -> str:
    """
    Creates the detailed prompt for the LLM,
    now including a 'riskWords' field in the JSON.
    """
    prompt = f"""
You are an expert call analyst. Based on the provided call transcript, generate a structured report containing the following sections:
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

# For consistent error fallback, ensure we have all 7 keys
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
        # --- Gemini API Call ---
        response = gemini_model.generate_content(prompt)

        # Access the generated text
        llm_response_content = response.text
        # --- End Gemini API Call ---

        if not llm_response_content:
            logger.error("Gemini LLM response content is empty.")
            return default_report

        # Clean potential code fences (Gemini might add them)
        cleaned_content = llm_response_content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3]

        # Parse the JSON
        report_data = json.loads(cleaned_content.strip())

        # Ensure all expected keys
        expected_keys = [
            "feedback", "keyTopics", "emotions", "sentiment", "output", "riskWords", "summary"
        ]
        for key in expected_keys:
            if key not in report_data:
                # Provide a default value or log a warning if a key is missing
                report_data[key] = report_data.get(key, f"Missing '{key}' in LLM response.")
                logger.warning(f"Key '{key}' missing in LLM JSON response. Using default.")


        return report_data

    except json.JSONDecodeError as json_err:
        logger.error(f"Failed to decode JSON response from LLM: {json_err}")
        logger.error(f"LLM raw response was: {llm_response_content}")
        return default_report
    except Exception as e:
        # Catch potential Gemini API errors (e.g., API key issues, rate limits)
        logger.error(f"Error generating call report with Gemini LLM: {e}", exc_info=True)
        # Return the default fallback
        return default_report


# ----------- Upload endpoint (no SSE, returns final JSON) -----------
@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Single endpoint:
      1) Save the file (optional).
      2) Hardcode or run Whisper for transcript.
      3) Check OCI for sentiment (optional).
      4) Ask Gemini LLM for a structured call report including 'riskWords'.
      5) Convert everything to your front-end's expected JSON shape and return it directly.
    """

    # 1) Save file to disk (optional)
    # Ensure temp directory exists if needed
    # temp_dir = "temp_audio"
    # os.makedirs(temp_dir, exist_ok=True)
    # temp_filename = os.path.join(temp_dir, f"temp_{uuid.uuid4()}_{os.path.splitext(file.filename)[0]}.wav")
    temp_filename = f"temp_{uuid.uuid4()}_{os.path.splitext(file.filename)[0]}.wav" # Simpler path

    try:
        with open(temp_filename, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"Saved uploaded file to {temp_filename}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save audio file: {e}")

    # 2) Transcribe with Whisper
    transcript_text = None # Initialize
    try:
        logger.info("Loading Whisper model...")
        # Consider using a smaller model if 'base' is too slow, or larger for accuracy
        model = whisper.load_model("base")
        logger.info("Starting transcription...")
        # Determine if GPU is available (fp16=True) or force CPU (fp16=False)
        # This example assumes CPU-only for broader compatibility
        result = model.transcribe(temp_filename, fp16=False)
        transcript_text = result["text"]
        logger.info("Whisper transcription complete.")
        if not transcript_text:
             logger.warning("Whisper returned an empty transcript.")
             # Use placeholder if empty, or raise error
             transcript_text = "[Transcription resulted in empty text]"
             # raise HTTPException(status_code=500, detail="Transcription failed: Empty result")
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}", exc_info=True)
        # Clean up the temp file even if transcription fails
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except OSError as rm_err:
                logger.warning(f"Could not remove temp file {temp_filename} after error: {rm_err}")
        raise HTTPException(status_code=500, detail=f"Whisper transcription failed: {e}")


    # --- Use hardcoded transcript for testing if needed ---
    # Uncomment the block below to bypass Whisper during development/testing
    """
    logger.warning("USING HARDCODED TRANSCRIPT TEXT FOR TESTING!")
    transcript_text = \"\"\"Alex:
    Good morning! This is Alex calling from Quick Tech Solutions. How are you doing today?

    Jamie:
    Are you seriously asking me that? After everything your company has put me through? I’ve spent hours on hold, been transferred a dozen times, and still have no solution.

    Alex:
    I'm really sorry to hear that, Jamie. I’d be happy to look into this and try to resolve it right away. Can you tell me a bit more about what happened?

    Jamie:
    What happened is I bought one of your so-called smart dishwashers and it broke after five days. Five. Days. Since then, I’ve been promised a replacement, a refund, a technician—you name it. None of that happened. I’ve been lied to, ignored, and completely disrespected as a customer.

    Alex:
    That’s absolutely not the experience we want our customers to have. Let me check your file and—

    Jamie:
    Don’t even bother reading off the same script everyone else has. I’ve heard it all. We're escalating it. We'll get back to you. Thank you for your patience. It’s insulting at this point. Just admit you sold me a broken product and don’t want to fix it.

    Alex:
    I understand you're upset, and you have every right to be. What you’ve experienced is unacceptable. I’ll personally make sure this gets handled today.

    Jamie:
    You said that last week. And the week before. Every time someone says they’re taking care of it, nothing happens. I’m done playing nice. If I don’t get a refund within 48 hours, I’m filing a complaint with the Better Business Bureau, disputing the charge with my bank, and leaving a detailed review everywhere I can.

    Alex:
    That’s completely understandable. Let me at least get the process moving again while we're on the call. I’ll also put in a high-priority note so your case gets addressed within the hour.

    Jamie:
    I’ll believe it when I see it. So far, all I’ve gotten from Quick Tech Solutions is broken promises and endless apologies that mean nothing.

    Alex:
    You shouldn’t have to go through that. I’m genuinely sorry. I’ll stay on top of this until you receive confirmation. You have my word.

    Jamie:
    Your word doesn’t mean anything to me unless I see results. I’ve wasted enough time on this disaster.

    Alex:
    Understood. You’ll hear back from me directly before the end of the day with a resolution.

    Jamie:
    Good. You’d better follow through this time.

    Alex:
    Thank you for your time, Jamie. I really do appreciate your patience.

    Jamie:
    I’m not being patient. I’m being very clear. Fix it.

    Alex:
    Got it. You’ll hear from me soon.

    Jamie:
    You’d better. Goodbye.
    \"\"\"
    """
    # --- End hardcoded transcript ---


    # 3) OCI Sentiment Analysis (optional)
    oci_emotion_text = "N/A"
    oci_aspects_list = []
    if language_client:
        try:
            documents = [
                TextDocument(
                    key="doc1",
                    text=transcript_text, # Use the actual transcript
                    language_code="en" # Or detect language if needed
                )
            ]
            # Ensure text is not empty before sending to OCI
            if transcript_text and transcript_text.strip() and transcript_text != "[Transcription resulted in empty text]":
                sentiment_request = BatchDetectLanguageSentimentsDetails(documents=documents)
                response = language_client.batch_detect_language_sentiments(
                    batch_detect_language_sentiments_details=sentiment_request
                )
                if response.data.documents and len(response.data.documents) > 0:
                    doc_result = response.data.documents[0]
                    # Extract sentiment label more robustly
                    oci_emotion_text = getattr(doc_result, 'document_sentiment', None)
                    if oci_emotion_text:
                        oci_emotion_text = oci_emotion_text.label
                    elif doc_result.aspects: # Fallback to first aspect sentiment if document sentiment missing
                         if doc_result.aspects[0].sentiment:
                             oci_emotion_text = doc_result.aspects[0].sentiment
                         else: oci_emotion_text = "Mixed (aspects found)"
                    else:
                        oci_emotion_text = "Neutral (default)" # Or Undetermined

                    if doc_result.aspects:
                        oci_aspects_list = [
                            {
                                "text": a.text,
                                "sentiment": a.sentiment,
                                "scores": a.scores # Be aware scores might be dict/object
                            } for a in doc_result.aspects
                        ]
            else:
                logger.warning("Skipping OCI sentiment analysis due to empty transcript.")
                oci_emotion_text = "N/A (empty transcript)"

        except Exception as e:
            logger.error(f"OCI sentiment analysis failed: {e}")
            oci_emotion_text = "Error (OCI failure)"

    # 4) Generate the structured LLM report using Gemini
    report_data = generate_call_report_with_llm(transcript_text)

    # 5) Save in temp storage (optional)
    transcript_id = str(uuid.uuid4())
    try:
        # Assuming save_transcript exists and works
        save_transcript(
            transcript_id=transcript_id,
            transcript_text=transcript_text,
            report_data=report_data,
            oci_emotion=oci_emotion_text,
            oci_aspects=oci_aspects_list
        )
        logger.info(f"Transcript data saved for ID: {transcript_id}")
    except NameError:
         logger.warning("`save_transcript` function not found in `temp_storage`. Skipping save.")
    except Exception as e:
        logger.error(f"Failed to save transcript data for {transcript_id}: {e}")

    # Remove the temp audio file
    if os.path.exists(temp_filename):
        try:
            os.remove(temp_filename)
            logger.info(f"Removed temp audio file: {temp_filename}")
        except OSError as e:
            logger.warning(f"Could not remove temp file {temp_filename}: {e}")

    # Convert transcript into an array of {speaker, text} to match your front-end
    parsed_transcript = parse_transcript_to_list(transcript_text)

    # Build final JSON in the same shape as your front-end FileData
    # We'll use the transcript_id as the unique ID (string)
    # Date can be dynamic
    current_date_str = date.today().strftime("%d/%m/%Y") # DD/MM/YYYY format

    final_response = {
        "id": transcript_id, # Use the generated UUID string
        "name": file.filename if file else "Unknown Filename",
        "date": current_date_str,
        "type": "Sin categoría", # Or derive from filename/metadata if possible
        "duration": "N/A", # TODO: Calculate duration if possible (e.g., using an audio library)
        "rating": 0, # Default rating
        "report": {
            # Use .get with defaults for safety, though generate_call_report_with_llm should handle defaults
            "feedback": report_data.get("feedback", "N/A"),
            "keyTopics": report_data.get("keyTopics", []),
            "emotions": report_data.get("emotions", []),
            "sentiment": report_data.get("sentiment", "N/A"),
            "output": report_data.get("output", "N/A"),
            "riskWords": report_data.get("riskWords", "N/A"),
            "summary": report_data.get("summary", "N/A")
        },
        "transcript": parsed_transcript
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

    try:
        # Assuming load_transcript exists and works
        transcript_obj = load_transcript(request.transcript_id)
    except NameError:
        logger.error("`load_transcript` function not found in `temp_storage`.")
        raise HTTPException(status_code=500, detail="Server configuration error: temp storage unavailable.")

    if not transcript_obj:
        logger.warning(f"Transcript not found for id: {request.transcript_id}")
        raise HTTPException(status_code=404, detail="Transcript not found")

    report = transcript_obj.get("report", {})
    transcript_text = transcript_obj.get("transcript_text", "Transcript not available.")
    oci_emotion = transcript_obj.get("oci_emotion", "N/A")

    # Build system prompt / context for Gemini
    # Gemini works well with clear instructions at the start.
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
        f"\n--- Transcript Snippet (start) ---\n{transcript_text[:1000]}...\n--- Transcript Snippet (end) ---", # Increased snippet size
        "\nBased *only* on the provided transcript information and the analysis report, answer the user's questions concisely. Do not invent information not present in the provided context."
    ]
    initial_context = "\n".join(context_parts)

    # Format message history for Gemini's generate_content
    # It expects a list of {'role': 'user'/'model', 'parts': ['text']}
    gemini_messages = []
    # Add the system context as the first 'user' message (or potentially a specific system role if API evolves)
    gemini_messages.append({'role': 'user', 'parts': [initial_context]})
    # Add a placeholder 'model' response to set the context for the actual user query
    gemini_messages.append({'role': 'model', 'parts': ["Okay, I have reviewed the call context. How can I help you?"]})

    # Append the actual conversation history
    for msg in request.messages:
        role = "model" if msg.get("role") == "assistant" else "user" # Gemini uses 'model' for assistant
        content = msg.get("content", "")
        if content: # Avoid adding empty messages
             gemini_messages.append({'role': role, 'parts': [content]})

    # Ensure the last message is from the 'user' for the request to make sense
    if not gemini_messages or gemini_messages[-1]['role'] != 'user':
         # This shouldn't happen if the frontend sends user questions, but handle defensively
         logger.warning("Chat request did not end with a user message. Appending a default prompt.")
         gemini_messages.append({'role': 'user', 'parts': ["Please provide information based on the context."]})


    try:
        logger.info(f"Sending chat request to Gemini LLM for transcript {request.transcript_id}")
        # --- Gemini API Call for Chat ---
        # Use generate_content with the message history
        response = gemini_model.generate_content(gemini_messages)

        assistant_reply = response.text
        # --- End Gemini API Call ---

        if not assistant_reply:
            logger.warning("Gemini LLM returned empty reply for chat request.")
            # Check for safety blocks, though .text usually handles this
            try:
                 print(response.prompt_feedback) # Log safety feedback if available
            except Exception:
                 pass # Ignore if prompt_feedback isn't present
            return {"assistant_message": "Sorry, I couldn't generate a response for that request."}

        return {"assistant_message": assistant_reply}

    except Exception as e:
        # Catch potential Gemini API errors
        logger.error(f"Exception calling Gemini LLM during chat for transcript {request.transcript_id}: {e}", exc_info=True)
        # Provide a more specific error if possible (e.g., check for common Gemini error types)
        # Example: Check if it's related to API key or quota
        error_detail = f"LLM service error: {str(e)}"
        if "API key not valid" in str(e):
            error_detail = "LLM service error: Invalid API Key."
        # Add more specific checks if needed

        raise HTTPException(status_code=503, detail=error_detail)


# ---------- Test OCI endpoint ----------
@app.get("/test_oci")
def test_oci_call():
    """
    Quick test route for checking OCI doc-level sentiment.
    """
    if not language_client:
        logger.error("OCI language client not configured.")
        return {"error": "OCI client not configured."}

    # Use a shorter, more representative text for testing
    text = "I am very unhappy with the service, it was slow and the product broke quickly. However, the support agent Alex was trying to be helpful."

    documents = [TextDocument(key="doc1", text=text, language_code="en")]
    sentiment_request = BatchDetectLanguageSentimentsDetails(documents=documents)

    try:
        logger.info("Sending test request to OCI Language Service...")
        response = language_client.batch_detect_language_sentiments(
            batch_detect_language_sentiments_details=sentiment_request
        )
        logger.info(f"/test_oci OCI response status: {response.status}")
        # Convert OCI models to dict for JSON serialization if needed
        # response.data might contain OCI model objects
        try:
            # Attempt direct return, FastAPI might handle it. If not, convert manually.
            return response.data
        except Exception:
             # Basic conversion if direct return fails
            return {"data": str(response.data)}
    except oci.exceptions.ServiceError as oci_err:
         logger.error(f"/test_oci OCI Service Error: Status {oci_err.status}, Code {oci_err.code}, Message {oci_err.message}", exc_info=True)
         return {"error": f"OCI Service Error: {oci_err.message} (Status: {oci_err.status})"}
    except Exception as ex:
        logger.error(f"/test_oci generic exception: {ex}", exc_info=True)
        return {"error": str(ex)}


# ---------- Run server if executed directly ----------
if __name__ == "__main__":
    # Check if Gemini key is set before starting
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
         print("--- WARNING: GEMINI_API_KEY is not set. LLM features will not work. ---")
    if gemini_model is None and (GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE"):
         print("--- WARNING: Gemini model failed to initialize despite API key being present. Check configuration and logs. ---")

    logger.info("Starting FastAPI server on host 0.0.0.0, port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)