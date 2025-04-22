# temp_storage.py
import os
import json
import logging

# Get a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Avoid adding duplicate handlers if root logger is already configured
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# Folder to store all temporary transcript files
STORAGE_DIR = "transcript_storage"

# Ensure the folder exists on startup or when the module is loaded
try:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    logger.info(f"Ensured transcript storage directory exists: {STORAGE_DIR}")
except OSError as e:
    # Handle potential permission issues or other errors during directory creation
    logger.error(f"Could not create storage directory {STORAGE_DIR}: {e}", exc_info=True)


def save_transcript(
    transcript_id: str,
    diarized_transcript: list, # List of dicts: {'speaker': str, 'text': str, 'start': float, 'end': float}
    full_transcript_text: str, # String: "SPEAKER_00: Hello...\nSPEAKER_01: Hi..."
    report_data: dict,         # Dict containing LLM analysis results
    oci_emotion: str,
    oci_aspects: list,
    **extra
):
    """
    Saves the given transcript data, including the structured diarized list
    (with speaker, text, start, end) and the full text string, to a JSON file.

    Args:
        transcript_id: Unique ID for the transcript.
        diarized_transcript: List of dictionaries with 'speaker', 'text', 'start', 'end'.
        full_transcript_text: The complete transcript as a single string with speaker labels.
        report_data: Dictionary containing the LLM analysis report.
        oci_emotion: Overall sentiment label from OCI.
        oci_aspects: List of aspect-level sentiment details from OCI.

    Raises:
        IOError: If saving the file fails.
        TypeError: If data cannot be serialized to JSON.
    """
    # Construct the data payload to be saved
    data_to_save: dict = {
        "id": transcript_id,
        "diarized_transcript": diarized_transcript, # Store the list with timestamps
        "full_transcript_text": full_transcript_text, # Store the full string
        "report_data": report_data,      # Store the LLM report
        "oci_emotion": oci_emotion,      # Store OCI overall sentiment
        "oci_aspects": oci_aspects       # Store OCI aspect sentiment
    }

    # Define the full path for the transcript file
    file_path = os.path.join(STORAGE_DIR, f"transcript_{transcript_id}.json")
    logger.info(f"Attempting to save transcript data to: {file_path}")

    try:
        # Write the data to the JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            # Use indent for readability, ensure_ascii=False for non-ASCII chars
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved transcript data for ID: {transcript_id} to {file_path}")

    except IOError as e:
        logger.error(f"IOError saving transcript {transcript_id} to {file_path}: {e}", exc_info=True)
        # Re-raise as IOError or a custom exception if needed
        raise IOError(f"Failed to write transcript file {transcript_id}") from e
    except TypeError as e:
        logger.error(f"TypeError serializing data for transcript {transcript_id}: {e}", exc_info=True)
        # This might happen if non-serializable objects are in the data
        raise TypeError(f"Data for transcript {transcript_id} is not JSON serializable") from e
    except Exception as e:
        # Catch any other unexpected errors during save
        logger.error(f"Unexpected error saving transcript {transcript_id} to {file_path}: {e}", exc_info=True)
        raise # Re-raise the original exception


def load_transcript(transcript_id: str) -> dict:
    """
    Retrieves the transcript JSON data by its transcript_id.

    Args:
        transcript_id: The unique ID of the transcript to load.

    Returns:
        A dictionary containing the loaded transcript data.

    Raises:
        FileNotFoundError: If no transcript file matches the given ID.
        ValueError: If the file content is not valid JSON.
        IOError: If reading the file fails for other reasons.
    """
    # Define the full path for the transcript file
    file_path = os.path.join(STORAGE_DIR, f"transcript_{transcript_id}.json")
    logger.info(f"Attempting to load transcript data from: {file_path}")

    # Check if the file exists before attempting to open it
    if not os.path.exists(file_path):
        logger.warning(f"Transcript file not found for ID: {transcript_id} at {file_path}")
        raise FileNotFoundError(f"Transcript data for ID '{transcript_id}' not found.")

    try:
        # Read the JSON data from the file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded transcript data for ID: {transcript_id} from {file_path}")
        return data

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from {file_path} for ID {transcript_id}: {e}", exc_info=True)
        # Raise a ValueError indicating corrupt or invalid file content
        raise ValueError(f"Invalid JSON format in transcript file for ID '{transcript_id}'.") from e
    except IOError as e:
         logger.error(f"IOError reading transcript file {file_path} for ID {transcript_id}: {e}", exc_info=True)
         # Re-raise as IOError or a custom exception
         raise IOError(f"Failed to read transcript file for ID '{transcript_id}'.") from e
    except Exception as e:
        # Catch any other unexpected errors during load
        logger.error(f"Unexpected error loading transcript {transcript_id} from {file_path}: {e}", exc_info=True)
        raise # Re-raise the original exception

