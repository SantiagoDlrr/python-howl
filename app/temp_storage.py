import os
import json

# Folder to store all temporary transcript files
TEMP_FOLDER = "temp_data"

# Ensure the folder exists on startup
os.makedirs(TEMP_FOLDER, exist_ok=True)

def save_transcript(
    transcript_id: str,
    transcript_text: str,
    report_data: dict,
    oci_emotion: str,
    oci_aspects: list
):
    """
    Saves the given transcript data to a JSON file using the provided transcript_id.
    Returns the same transcript_id for convenience.
    """
    data = {
        "id": transcript_id,
        "transcript_text": transcript_text,
        "report": report_data,      # Full LLM-generated report
        "oci_emotion": oci_emotion, # Overall sentiment label from OCI
        "oci_aspects": oci_aspects  # Aspect-level sentiment from OCI
    }

    file_path = os.path.join(TEMP_FOLDER, f"transcript_{transcript_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return transcript_id

def load_transcript(transcript_id: str):
    """
    Retrieves the transcript JSON by its transcript_id.
    Returns None if no matching file is found.
    """
    file_path = os.path.join(TEMP_FOLDER, f"transcript_{transcript_id}.json")
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
