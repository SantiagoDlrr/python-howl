# # transcription_module.py
# import whisperx
# import gc
# import torch
# import os
# import logging
# from typing import List, Dict, Optional

# # Configure logging for this module
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)


# class AudioTranscriber:
#     """
#     Handles audio transcription and speaker diarization using whisperX.
#     """
#     def __init__(self,
#                  model_name: str = "base",
#                  device: str = "cpu",
#                  compute_type: str = "int8", # float16 for GPU, int8 for CPU (faster, lower quality)
#                  hf_token: Optional[str] = None):
#         """
#         Initializes the transcription and diarization models.

#         Args:
#             model_name: Name of the Whisper model (e.g., "tiny", "base", "small", "medium", "large-v2").
#             device: Device to run inference on ("cpu" or "cuda").
#             compute_type: Quantization type ("float16", "int8", etc.).
#             hf_token: Hugging Face token for pyannote.audio. Reads from env HF_TOKEN if None.
#         """
#         self.device = device
#         self.compute_type = compute_type
#         self.model_name = model_name
#         self.hf_token = hf_token or os.getenv("HF_TOKEN")

#         if self.device == "cuda" and not torch.cuda.is_available():
#             logger.warning("CUDA specified but not available. Falling back to CPU.")
#             self.device = "cpu"
#             self.compute_type = "int8" # Recommended for CPU

#         if not self.hf_token:
#              logger.warning("Hugging Face token (HF_TOKEN) not found in environment variables. "
#                             "Diarization might fail if pyannote model requires authentication.")
#              # Raise error might be better depending on requirements
#              # raise ValueError("Hugging Face token is required for diarization.")


#         logger.info(f"Initializing AudioTranscriber with model='{model_name}', device='{self.device}', compute_type='{self.compute_type}'")
#         try:
#             self.model = whisperx.load_model(self.model_name, self.device, compute_type=self.compute_type)
#             logger.info(f"WhisperX model '{model_name}' loaded successfully.")
#             # Diarization model loaded on demand in transcribe_and_diarize
#             self.diarize_model = None
#         except Exception as e:
#             logger.error(f"Failed to load WhisperX model '{model_name}': {e}", exc_info=True)
#             raise RuntimeError(f"Failed to initialize AudioTranscriber model: {e}") from e


#     def _load_diarize_model(self):
#         """Loads the diarization model if not already loaded."""
#         if self.diarize_model is None:
#             logger.info("Loading diarization model...")
#             try:
#                 # Check token here again if needed, though whisperx might handle it
#                 if not self.hf_token:
#                      logger.error("Cannot load diarization model without Hugging Face token.")
#                      # Return None or raise error to indicate failure
#                      return None # Indicate failure to load
#                 self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
#                 logger.info("Diarization model loaded successfully.")
#                 return self.diarize_model
#             except Exception as e:
#                 logger.error(f"Failed to load diarization model: {e}", exc_info=True)
#                 # Propagate failure
#                 return None
#         return self.diarize_model

#     def transcribe_and_diarize(self, audio_path: str, batch_size: int = 16, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None) -> List[Dict[str, str]]:
#         """
#         Transcribes audio and performs speaker diarization.

#         Args:
#             audio_path: Path to the audio file.
#             batch_size: Batch size for transcription inference.
#             min_speakers: Minimum number of speakers expected.
#             max_speakers: Maximum number of speakers expected.

#         Returns:
#             A list of dictionaries, where each dictionary represents a speaker segment:
#             [{"speaker": "SPEAKER_XX", "text": "Utterance text..."}, ...]
#             Returns an empty list or list with error message on failure.
#         """
#         logger.info(f"Starting transcription for: {audio_path}")
#         try:
#             # 1. Transcribe with whisperX (gives word timestamps)
#             audio = whisperx.load_audio(audio_path)
#             # Need to handle potential memory issues with large files on CPU
#             result = self.model.transcribe(audio, batch_size=batch_size)
#             logger.info("Initial transcription complete.")

#             if not result or not result.get("segments"):
#                 logger.warning("Transcription returned no segments.")
#                 return [{"speaker": "SYSTEM", "text": "[Transcription failed or audio was empty]"}]

#              # 2. Align whisper output
#             logger.info("Aligning transcript...")
#             # Check if model_a and metadata are already loaded or need loading here
#             # This might vary based on whisperx version, ensure alignment model is available
#             try:
#                  model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
#                  aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
#                  logger.info("Alignment complete.")
#                  del model_a # Free memory
#                  gc.collect()
#                  torch.cuda.empty_cache() if self.device == 'cuda' else None
#             except Exception as align_err:
#                  logger.error(f"Failed during alignment: {align_err}", exc_info=True)
#                  # Fallback: Use original segments if alignment fails? Or return error?
#                  # Let's try to continue with unaligned segments for basic diarization, but log warning
#                  logger.warning("Proceeding with potentially less accurate unaligned segments for diarization.")
#                  aligned_result = {"segments": result["segments"], "word_segments": []} # Mock structure if needed


#             # 3. Load diarization model (if not already loaded)
#             diarize_model = self._load_diarize_model()
#             if diarize_model is None:
#                  logger.error("Diarization model could not be loaded. Skipping diarization.")
#                  # Return transcript without speaker labels (maybe formatted differently)
#                  # Or return an error indicator
#                  # For now, let's return the raw text concatenated
#                  raw_text = " ".join([seg['text'].strip() for seg in result.get("segments", [])])
#                  return [{"speaker": "UNKNOWN", "text": f"[Diarization failed] {raw_text}"}]

#             # 4. Perform Speaker Diarization
#             logger.info("Performing speaker diarization...")
#             try:
#                 diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
#                 # Ensure diarize_segments has expected structure (can be pd.DataFrame or dict)
#                 # whisperx expects specific format, might need adjustment if pyannote output changed
#                 logger.info("Diarization complete.")
#             except Exception as diarize_err:
#                 logger.error(f"Failed during diarization pipeline execution: {diarize_err}", exc_info=True)
#                 raw_text = " ".join([seg['text'].strip() for seg in result.get("segments", [])])
#                 return [{"speaker": "UNKNOWN", "text": f"[Diarization failed] {raw_text}"}]


#             # 5. Assign speaker labels to word segments
#             # Check if 'word_segments' exists, which comes from alignment
#             if "word_segments" not in aligned_result or not aligned_result["word_segments"]:
#                  logger.warning("No word segments found after alignment. Cannot assign speakers accurately.")
#                  # Fallback: Assign speakers based on segment-level timings (less accurate)
#                  # Or return the unlabelled transcript
#                  raw_text = " ".join([seg['text'].strip() for seg in result.get("segments", [])])
#                  return [{"speaker": "UNKNOWN", "text": f"[Speaker assignment failed due to missing word timings] {raw_text}"}]

#             logger.info("Assigning speakers to words...")
#             try:
#                 result_segments_with_speakers = whisperx.assign_word_speakers(diarize_segments, aligned_result)
#                 logger.info("Speaker assignment complete.")
#             except Exception as assign_err:
#                  logger.error(f"Failed during speaker assignment: {assign_err}", exc_info=True)
#                  raw_text = " ".join([seg['text'].strip() for seg in result.get("segments", [])])
#                  return [{"speaker": "UNKNOWN", "text": f"[Speaker assignment failed] {raw_text}"}]

#             # 6. Format the output
#             formatted_transcript = self._format_output(result_segments_with_speakers["segments"])

#             # Cleanup GPU memory
#             if self.device == 'cuda':
#                  del result
#                  del aligned_result
#                  # Optionally del diarize_model if not needed immediately after
#                  # del self.diarize_model
#                  # self.diarize_model = None
#                  gc.collect()
#                  torch.cuda.empty_cache()

#             logger.info("Transcription and diarization process finished.")
#             return formatted_transcript

#         except Exception as e:
#             logger.error(f"An error occurred during transcription/diarization for {audio_path}: {e}", exc_info=True)
#              # Cleanup GPU memory on error too
#             if self.device == 'cuda':
#                 gc.collect()
#                 torch.cuda.empty_cache()
#             return [{"speaker": "SYSTEM", "text": f"[Error during processing: {e}]"}]


#     def _format_output(self, segments: List[Dict]) -> List[Dict[str, str]]:
#         """
#         Formats the diarized segments into the desired list of speaker turns.
#         Groups consecutive segments from the same speaker.
#         """
#         if not segments:
#             return []

#         output: List[Dict[str, str]] = []
#         current_speaker = None
#         current_text = []

#         logger.debug(f"Raw segments for formatting: {segments}")

#         for segment in segments:
#             # whisperX usually adds 'speaker' key after assign_word_speakers
#             speaker_label = segment.get('speaker', 'UNKNOWN')
#             text = segment.get('text', '').strip()

#             if not text: # Skip empty segments
#                 continue

#             if current_speaker is None: # First segment
#                 current_speaker = speaker_label
#                 current_text.append(text)
#             elif speaker_label == current_speaker: # Same speaker continues
#                 current_text.append(text)
#             else: # Speaker changed
#                 # Save the previous speaker's turn
#                 if current_text:
#                     output.append({"speaker": current_speaker, "text": " ".join(current_text)})
#                 # Start the new speaker's turn
#                 current_speaker = speaker_label
#                 current_text = [text]

#         # Add the last speaker's turn
#         if current_speaker is not None and current_text:
#             output.append({"speaker": current_speaker, "text": " ".join(current_text)})

#         logger.info(f"Formatted transcript into {len(output)} speaker turns.")
#         return output




#     def get_full_text(self, diarized_transcript: List[Dict[str, str]]) -> str:
#         """
#         Helper function to get the full transcript text string with speaker labels.
#         """
#         return "\n".join([f"{entry['speaker']}: {entry['text']}" for entry in diarized_transcript])


# # Example usage (optional, for testing the module directly)
# if __name__ == "__main__":
#     print("Testing transcription module...")
#     # Make sure HF_TOKEN is set in your environment
#     # You need a sample audio file (e.g., sample.wav)
#     if not os.getenv("HF_TOKEN"):
#         print("ERROR: HF_TOKEN environment variable not set. Diarization will likely fail.")
#         exit(1)

#     if not os.path.exists("sample.wav"):
#         print("ERROR: sample.wav not found for testing.")
#         print("Please place a sample audio file named 'sample.wav' in the same directory.")
#         # Create a dummy silent file for basic testing if ffmpeg is available
#         try:
#             print("Attempting to create a dummy silent sample.wav using ffmpeg...")
#             os.system("ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -t 5 -q:a 9 sample.wav")
#             if not os.path.exists("sample.wav"):
#                  raise FileNotFoundError("ffmpeg failed to create dummy file.")
#             print("Dummy sample.wav created.")
#         except Exception as e:
#             print(f"Could not create dummy sample file: {e}. Please provide a real sample.wav.")
#             exit(1)


#     # Determine device
#     test_device = "cuda" if torch.cuda.is_available() else "cpu"
#     test_compute = "float16" if test_device == "cuda" else "int8"

#     try:
#         transcriber = AudioTranscriber(model_name="tiny", device=test_device, compute_type=test_compute) # Use tiny for faster testing
#         print(f"Transcriber initialized on {test_device}.")
#         diarized_result = transcriber.transcribe_and_diarize("sample.wav")

#         print("\n--- Diarized Transcript (List of Dicts) ---")
#         for item in diarized_result:
#             print(item)

#         print("\n--- Full Text String ---")
#         full_text = transcriber.get_full_text(diarized_result)
#         print(full_text)

#     except RuntimeError as e:
#         print(f"\nError during testing: {e}")
#     except Exception as e:
#         print(f"\nAn unexpected error occurred during testing: {e}")

#     # Clean up dummy file
#     if os.path.exists("sample.wav") and "anullsrc" in open("sample.wav", "rb").read(100).decode('latin-1', errors='ignore'): # Basic check if it's the dummy
#         try:
#             os.remove("sample.wav")
#             print("Dummy sample.wav removed.")
#         except OSError:

#             print("Could not remove dummy sample.wav.")





# transcription_module.py
import whisperx
import gc
import torch
import os
import logging
from typing import List, Dict, Optional, Any

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Avoid adding duplicate handlers if root logger is already configured
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class AudioTranscriber:
    """
    Handles audio transcription and speaker diarization using whisperX.
    """
    def __init__(self,
                 model_name: str = "base",
                 device: str = "cpu",
                 compute_type: str = "int8", # float16 for GPU, int8 for CPU
                 hf_token: Optional[str] = None):
        """
        Initializes the transcription and diarization models.

        Args:
            model_name: Name of the Whisper model (e.g., "tiny", "base", "small", "medium", "large-v2").
            device: Device to run inference on ("cpu" or "cuda").
            compute_type: Quantization type ("float16", "int8", etc.).
            hf_token: Hugging Face token for pyannote.audio. Reads from env HF_TOKEN if None.
        """
        self.device = device
        self.compute_type = compute_type
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA specified but not available. Falling back to CPU.")
            self.device = "cpu"
            # Ensure compute_type is suitable for CPU if falling back
            if self.compute_type not in ["int8", "float32"]: # Add other valid CPU types if needed
                 self.compute_type = "int8"
                 logger.warning(f"Compute type adjusted to '{self.compute_type}' for CPU fallback.")


        if not self.hf_token:
             logger.warning("Hugging Face token (HF_TOKEN) not found in environment variables. "
                            "Diarization will fail if pyannote model requires authentication or accepting terms.")
             # Consider raising error depending on requirements:
             # raise ValueError("Hugging Face token is required for diarization.")

        logger.info(f"Initializing AudioTranscriber with model='{self.model_name}', device='{self.device}', compute_type='{self.compute_type}'")
        try:
            # Load the main Whisper model
            self.model = whisperx.load_model(self.model_name, self.device, compute_type=self.compute_type)
            logger.info(f"WhisperX model '{self.model_name}' loaded successfully.")

            # Diarization model is loaded lazily in _load_diarize_model to handle potential token issues gracefully
            self.diarize_model = None
            # Alignment model is also loaded on demand during transcription
            self.align_model = None
            self.align_metadata = None

        except Exception as e:
            logger.error(f"Failed to load WhisperX model '{self.model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize AudioTranscriber base model: {e}") from e

    def _load_diarize_model(self):
        """Loads the diarization model if not already loaded."""
        if self.diarize_model is None:
            logger.info("Attempting to load diarization model...")
            if not self.hf_token:
                 logger.error("Cannot load diarization model: Hugging Face token (HF_TOKEN) is missing.")
                 return None # Indicate failure clearly

            try:
                # Use the provided token for authentication
                self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
                logger.info("Diarization model loaded successfully.")
                return self.diarize_model
            except Exception as e:
                logger.error(f"Failed to load diarization model: {e}", exc_info=True)
                logger.error("This often means the HF_TOKEN is invalid OR you haven't accepted the model terms "
                             "on Hugging Face Hub (e.g., for pyannote/segmentation-3.0).")
                # Set to None to prevent retries within the same transcription call
                self.diarize_model = None
                return None # Indicate failure
        return self.diarize_model

    def _load_align_model(self, language_code: str):
        """Loads the alignment model for the detected language if not already loaded."""
        # Check if model for this language is already loaded
        if self.align_model is not None and self.align_metadata is not None and self.align_metadata.get("language_code") == language_code:
            logger.debug(f"Using existing alignment model for language: {language_code}")
            return self.align_model, self.align_metadata

        logger.info(f"Loading alignment model for language: {language_code}...")
        try:
            # Unload previous model if language changed to free memory (optional but good practice)
            if self.align_model is not None:
                del self.align_model
                del self.align_metadata
                self.align_model = None
                self.align_metadata = None
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

            model_a, metadata = whisperx.load_align_model(language_code=language_code, device=self.device)
            self.align_model = model_a
            # Store language code in metadata for checking later
            metadata["language_code"] = language_code
            self.align_metadata = metadata
            logger.info(f"Alignment model for language {language_code} loaded successfully.")
            return self.align_model, self.align_metadata
        except Exception as e:
            logger.error(f"Failed to load alignment model for language {language_code}: {e}", exc_info=True)
            # Reset state
            self.align_model = None
            self.align_metadata = None
            return None, None # Indicate failure

    def transcribe_and_diarize(self, audio_path: str, batch_size: int = 16, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Transcribes audio and performs speaker diarization with timestamps.

        Args:
            audio_path: Path to the audio file.
            batch_size: Batch size for transcription inference.
            min_speakers: Minimum number of speakers expected for diarization.
            max_speakers: Maximum number of speakers expected for diarization.

        Returns:
            A list of dictionaries, where each dictionary represents a speaker segment:
            [{"speaker": "SPEAKER_XX", "text": "Utterance text...", "start": 0.0, "end": 0.0}, ...]
            Returns an empty list or list with error message on failure.
        """
        logger.info(f"Starting transcription and diarization process for: {audio_path}")
        diarize_failed = False # Flag to track if diarization step fails

        try:
            # 1. Load audio
            try:
                 audio = whisperx.load_audio(audio_path)
            except Exception as load_err:
                 logger.error(f"Failed to load audio file {audio_path}: {load_err}", exc_info=True)
                 return [{"speaker": "SYSTEM", "text": f"[Error loading audio file: {load_err}]", "start": 0.0, "end": 0.0}]

            # 2. Transcribe with whisperX
            logger.info("Performing initial transcription...")
            result = self.model.transcribe(audio, batch_size=batch_size)
            detected_language = result.get("language")
            logger.info(f"Initial transcription complete. Detected language: {detected_language}")

            if not result or not result.get("segments"):
                logger.warning("Transcription returned no segments.")
                return [{"speaker": "SYSTEM", "text": "[Transcription failed or audio was empty]", "start": 0.0, "end": 0.0}]

             # 3. Align whisper output
            logger.info("Aligning transcript...")
            model_a, metadata = self._load_align_model(language_code=detected_language)
            if model_a is None or metadata is None:
                 # Alignment model failed to load, cannot proceed accurately
                 logger.error("Alignment model failed to load. Cannot perform alignment or diarization accurately.")
                 # Return basic transcript without timestamps or speakers
                 raw_text = " ".join([seg['text'].strip() for seg in result.get("segments", [])])
                 return [{"speaker": "UNKNOWN", "text": f"[Alignment Failed] {raw_text}", "start": 0.0, "end": 0.0}]

            try:
                 aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
                 logger.info("Alignment complete.")
                 # No need to delete model_a here as it's stored in self.align_model
                 # gc.collect() might still be useful if memory is tight
                 # torch.cuda.empty_cache() if self.device == 'cuda' else None
            except Exception as align_err:
                 logger.error(f"Failed during alignment execution: {align_err}", exc_info=True)
                 logger.warning("Proceeding without word timings. Diarization might be inaccurate or skipped.")
                 # Create a structure mimicking aligned_result but indicating failure/missing words
                 aligned_result = {"segments": result["segments"], "word_segments": []}


            # 4. Load and Perform Speaker Diarization
            diarize_model = self._load_diarize_model()
            if diarize_model is None:
                 logger.warning("Diarization model could not be loaded or failed. Skipping speaker assignment.")
                 diarize_failed = True
                 # Proceed without speaker labels, using original segments
                 result_segments_with_speakers = aligned_result # Use aligned result for timestamps if available
            else:
                logger.info("Performing speaker diarization...")
                try:
                    diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
                    # Example: whisperx assumes specific format, might need adjustment if pyannote output changes
                    logger.info("Diarization complete.")

                    # 5. Assign speaker labels to word segments
                    # Check if 'word_segments' exists and is populated, essential for accurate assignment
                    if "word_segments" not in aligned_result or not aligned_result["word_segments"]:
                         logger.warning("No word segments found after alignment. Cannot assign speakers accurately.")
                         # Fallback: Use segment-level assignment (less accurate) or skip
                         # For simplicity, we'll flag diarization as failed if word segments are missing
                         diarize_failed = True
                         result_segments_with_speakers = aligned_result # Keep aligned result for timestamps
                    else:
                        logger.info("Assigning speakers to words...")
                        try:
                            result_segments_with_speakers = whisperx.assign_word_speakers(diarize_segments, aligned_result)
                            logger.info("Speaker assignment complete.")
                        except Exception as assign_err:
                             logger.error(f"Failed during speaker assignment: {assign_err}", exc_info=True)
                             diarize_failed = True
                             result_segments_with_speakers = aligned_result # Fallback to aligned result

                except Exception as diarize_err:
                    logger.error(f"Failed during diarization pipeline execution: {diarize_err}", exc_info=True)
                    diarize_failed = True
                    result_segments_with_speakers = aligned_result # Fallback to aligned result

            # 6. Format the output
            # Pass the segments (potentially with speaker info) to the formatting function
            formatted_transcript = self._format_output(result_segments_with_speakers.get("segments", []), diarize_failed)

            # Cleanup general memory (optional, depends on usage pattern)
            del result
            del aligned_result
            # Don't del diarize_model or align_model if you want them cached globally
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            logger.info("Transcription and diarization process finished.")
            return formatted_transcript

        except Exception as e:
            logger.error(f"An critical error occurred during transcription/diarization for {audio_path}: {e}", exc_info=True)
             # Cleanup GPU memory on critical error too
            if self.device == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()
            return [{"speaker": "SYSTEM", "text": f"[Critical Error during processing: {e}]", "start": 0.0, "end": 0.0}]


    def _format_output(self, segments: List[Dict], diarize_failed: bool = False) -> List[Dict[str, Any]]:
        """
        Formats the diarized or aligned segments into the desired list of speaker turns,
        including start and end timestamps for each turn.
        Groups consecutive segments from the same speaker if diarization succeeded.

        Args:
            segments: List of segments from whisperx alignment or diarization.
                      Expected keys: 'text', 'start'/'end' (segment/word level), optional 'speaker'.
            diarize_failed: Boolean indicating if speaker assignment should be skipped.

        Returns:
            List of dictionaries: [{"speaker": str, "text": str, "start": float, "end": float}, ...]
        """
        if not segments:
            return []

        output: List[Dict[str, Any]] = []
        current_speaker = "UNKNOWN" if diarize_failed else None # Start as UNKNOWN if diarization failed
        current_text = []
        current_start_time = None
        current_end_time = None

        logger.debug(f"Formatting segments. Diarization failed: {diarize_failed}")

        for i, segment in enumerate(segments):
            # Determine speaker label
            # If diarization failed, all segments belong to 'UNKNOWN'
            # Otherwise, get label from segment or default to 'UNKNOWN' if missing
            speaker_label = "UNKNOWN" if diarize_failed else segment.get('speaker', 'UNKNOWN')
            text = segment.get('text', '').strip()
            words = segment.get('words') # Get word-level details for timestamps

            if not text: # Skip empty segments
                continue

            segment_start_time = None
            segment_end_time = None

            # Try to get timestamps from word level first (more accurate)
            if words and isinstance(words, list) and len(words) > 0:
                valid_words = [w for w in words if 'start' in w and 'end' in w and isinstance(w['start'], (int, float)) and isinstance(w['end'], (int, float))]
                if valid_words:
                    segment_start_time = valid_words[0].get('start')
                    segment_end_time = valid_words[-1].get('end')
                else:
                     logger.debug(f"Segment {i} has 'words' but no valid numeric start/end times within words.")

            # Fallback to segment-level timestamps if word level failed or wasn't present
            if segment_start_time is None and 'start' in segment and isinstance(segment['start'], (int, float)):
                segment_start_time = segment.get('start')
            if segment_end_time is None and 'end' in segment and isinstance(segment['end'], (int, float)):
                segment_end_time = segment.get('end')

            # Handle cases where timestamps might still be missing or invalid
            if segment_start_time is None:
                logger.warning(f"Segment {i} missing valid start time. Defaulting based on context.")
                segment_start_time = current_end_time if current_end_time is not None else 0.0
            if segment_end_time is None:
                 logger.warning(f"Segment {i} missing valid end time. Defaulting to start time + small duration.")
                 # Avoid end time being same as start if possible, add arbitrary small duration
                 segment_end_time = float(segment_start_time) + 0.1

            # Ensure conversion to float
            try:
                segment_start_time = float(segment_start_time)
                segment_end_time = float(segment_end_time)
                 # Basic sanity check: end time should not be before start time
                if segment_end_time < segment_start_time:
                    logger.warning(f"Correcting segment {i}: end time ({segment_end_time}) is before start time ({segment_start_time}). Setting end = start.")
                    segment_end_time = segment_start_time
            except (ValueError, TypeError) as time_err:
                 logger.error(f"Could not convert segment times to float for segment {i}. Start: {segment_start_time}, End: {segment_end_time}. Error: {time_err}. Skipping segment.")
                 continue # Skip this segment if times are invalid

            # --- Logic for grouping speaker turns ---
            # If diarization failed, treat every segment as a separate turn from "UNKNOWN"
            if diarize_failed:
                 output.append({
                    "speaker": "UNKNOWN",
                    "text": text,
                    "start": round(segment_start_time, 3),
                    "end": round(segment_end_time, 3)
                 })
                 # Update current_end_time for next segment's potential default start time
                 current_end_time = segment_end_time
                 continue # Move to the next segment

            # --- Normal grouping logic (diarization succeeded) ---
            if current_speaker is None: # First segment
                current_speaker = speaker_label
                current_text.append(text)
                current_start_time = segment_start_time
                current_end_time = segment_end_time
            elif speaker_label == current_speaker: # Same speaker continues
                current_text.append(text)
                # Only update the end time for the combined turn
                current_end_time = segment_end_time
            else: # Speaker changed
                # Save the previous speaker's turn if it has text and valid time
                if current_text and current_start_time is not None and current_end_time is not None:
                    output.append({
                        "speaker": current_speaker,
                        "text": " ".join(current_text).strip(),
                        "start": round(current_start_time, 3),
                        "end": round(current_end_time, 3)
                    })
                # Start the new speaker's turn
                current_speaker = speaker_label
                current_text = [text]
                current_start_time = segment_start_time
                current_end_time = segment_end_time

        # Add the last speaker's turn if diarization didn't fail
        if not diarize_failed and current_speaker is not None and current_text and current_start_time is not None and current_end_time is not None:
            output.append({
                "speaker": current_speaker,
                "text": " ".join(current_text).strip(),
                "start": round(current_start_time, 3),
                "end": round(current_end_time, 3)
            })

        logger.info(f"Formatted transcript into {len(output)} turns. Diarization failed: {diarize_failed}")
        return output

    def get_full_text(self, diarized_transcript: List[Dict[str, Any]]) -> str:
        """
        Helper function to get the full transcript text string with speaker labels
        from the formatted transcript list.

        Args:
            diarized_transcript: The list returned by _format_output.

        Returns:
            A single string concatenating speaker: text for each entry.
        """
        return "\n".join([f"{entry.get('speaker', 'UNKNOWN')}: {entry.get('text', '')}" for entry in diarized_transcript])