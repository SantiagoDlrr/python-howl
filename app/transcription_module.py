# """
# transcription_module.py
# -------------------------------------------------
# WhisperX wrapper for transcription, alignment, and optional diarization.

# Public use:

#     transcriber = AudioTranscriber(...)
#     diarized = transcriber.transcribe_and_diarize("path/to/audio")
#     full_text = transcriber.get_full_text(diarized)

# No knowledge of the runtime settings is required here.
# """
# from __future__ import annotations

# import os
# import gc
# import logging
# from typing import List, Dict, Optional, Any

# import torch
# import whisperx

# # --------------------------------------------------------------------------- #
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# if not logger.hasHandlers():
#     _h = logging.StreamHandler()
#     _h.setFormatter(
#         logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     )
#     logger.addHandler(_h)

# # --------------------------------------------------------------------------- #
# class AudioTranscriber:
#     """
#     Transcribe audio with WhisperX, then align and (optionally) diarize.
#     """
#     def __init__(
#         self,
#         model_name: str = "base",
#         device: str = "cpu",
#         compute_type: str = "int8",
#         hf_token: Optional[str] = None,
#     ):
#         self.device = device
#         self.compute_type = compute_type
#         self.model_name = model_name
#         self.hf_token = hf_token or os.getenv("HF_TOKEN")

#         if self.device == "cuda" and not torch.cuda.is_available():
#             logger.warning("CUDA requested but unavailable – falling back to CPU.")
#             self.device = "cpu"
#             self.compute_type = "int8"

#         if not self.hf_token:
#             logger.warning(
#                 "HF_TOKEN not set; diarization will fail if pyannote requires auth."
#             )

#         logger.info(
#             f"Loading WhisperX '{self.model_name}' on {self.device} "
#             f"(compute={self.compute_type})"
#         )
#         try:
#             self.model = whisperx.load_model(
#                 self.model_name, self.device, compute_type=self.compute_type
#             )
#         except Exception as e:
#             logger.error("WhisperX load failed", exc_info=True)
#             raise RuntimeError(f"WhisperX init failed: {e}") from e

#         self.diarize_model = None
#         self.align_model = None
#         self.align_metadata = None

#     # --------------------------------------------------------------------- #
#     def _load_diarize_model(self):
#         if self.diarize_model:
#             return self.diarize_model
#         if not self.hf_token:
#             logger.error("HF_TOKEN missing – diarization disabled.")
#             return None
#         try:
#             self.diarize_model = whisperx.DiarizationPipeline(
#                 use_auth_token=self.hf_token, device=self.device
#             )
#             return self.diarize_model
#         except Exception as e:
#             logger.error("Diarization load failed", exc_info=True)
#             self.diarize_model = None
#             return None

#     # --------------------------------------------------------------------- #
#     def _load_align_model(self, lang: str):
#         if (
#             self.align_model
#             and self.align_metadata
#             and self.align_metadata.get("language_code") == lang
#         ):
#             return self.align_model, self.align_metadata
#         try:
#             if self.align_model:
#                 del self.align_model, self.align_metadata
#                 if self.device == "cuda":
#                     torch.cuda.empty_cache()
#             model_a, metadata = whisperx.load_align_model(
#                 language_code=lang, device=self.device
#             )
#             metadata["language_code"] = lang
#             self.align_model, self.align_metadata = model_a, metadata
#             return model_a, metadata
#         except Exception as e:
#             logger.error("Alignment load failed", exc_info=True)
#             self.align_model = self.align_metadata = None
#             return None, None

#     # --------------------------------------------------------------------- #
#     def transcribe_and_diarize(
#         self,
#         audio_path: str,
#         batch_size: int = 16,
#         min_speakers: Optional[int] = None,
#         max_speakers: Optional[int] = None,
#     ) -> List[Dict[str, Any]]:
#         logger.info(f"Transcribing {audio_path}")
#         diarize_failed = False
#         try:
#             audio = whisperx.load_audio(audio_path)
#         except Exception as e:
#             logger.error("Audio load failed", exc_info=True)
#             return [{
#                 "speaker": "SYSTEM",
#                 "text": f"[Error loading audio: {e}]",
#                 "start": 0.0,
#                 "end": 0.0,
#             }]

#         result = self.model.transcribe(audio, batch_size=batch_size)
#         if not (result and result.get("segments")):
#             logger.warning("No segments returned.")
#             return [{
#                 "speaker": "SYSTEM",
#                 "text": "[Transcription failed]",
#                 "start": 0.0,
#                 "end": 0.0,
#             }]

#         lang = result.get("language")
#         model_a, metadata = self._load_align_model(lang)
#         if not model_a:
#             raw_text = " ".join(s["text"].strip() for s in result["segments"])
#             return [{
#                 "speaker": "UNKNOWN",
#                 "text": f"[Alignment failed] {raw_text}",
#                 "start": 0.0,
#                 "end": 0.0,
#             }]

#         aligned = whisperx.align(
#             result["segments"], model_a, metadata, audio, self.device,
#             return_char_alignments=False
#         )

#         diarizer = self._load_diarize_model()
#         if not diarizer:
#             diarize_failed = True
#             result_with_spk = aligned
#         else:
#             try:
#                 diar_segments = diarizer(
#                     audio, min_speakers=min_speakers, max_speakers=max_speakers
#                 )
#                 if not aligned.get("word_segments"):
#                     diarize_failed = True
#                     result_with_spk = aligned
#                 else:
#                     result_with_spk = whisperx.assign_word_speakers(
#                         diar_segments, aligned
#                     )
#             except Exception:
#                 logger.error("Diarization failed", exc_info=True)
#                 diarize_failed = True
#                 result_with_spk = aligned

#         formatted = self._format_output(
#             result_with_spk.get("segments", []), diarize_failed
#         )
#         gc.collect()
#         if self.device == "cuda":
#             torch.cuda.empty_cache()
#         return formatted

#     # --------------------------------------------------------------------- #
#     def _format_output(
#         self, segments: List[Dict], diarize_failed: bool
#     ) -> List[Dict[str, Any]]:
#         if not segments:
#             return []
#         out: List[Dict[str, Any]] = []
#         cur_spk = "UNKNOWN" if diarize_failed else None
#         cur_text, cur_start, cur_end = [], None, None

#         for seg in segments:
#             spk = "UNKNOWN" if diarize_failed else seg.get("speaker", "UNKNOWN")
#             txt = seg.get("text", "").strip()
#             if not txt:
#                 continue
#             s_start = (
#                 seg.get("words", [{}])[0].get("start")
#                 if seg.get("words") else seg.get("start")
#             )
#             s_end = (
#                 seg.get("words", [{}])[-1].get("end")
#                 if seg.get("words") else seg.get("end")
#             )
#             try:
#                 s_start, s_end = float(s_start), float(s_end)
#             except (TypeError, ValueError):
#                 s_start, s_end = 0.0, 0.0

#             if diarize_failed:
#                 out.append({
#                     "speaker": "UNKNOWN",
#                     "text": txt,
#                     "start": round(s_start, 3),
#                     "end": round(s_end, 3),
#                 })
#                 continue

#             if cur_spk is None:
#                 cur_spk, cur_text = spk, [txt]
#                 cur_start, cur_end = s_start, s_end
#             elif spk == cur_spk:
#                 cur_text.append(txt)
#                 cur_end = s_end
#             else:
#                 out.append({
#                     "speaker": cur_spk,
#                     "text": " ".join(cur_text).strip(),
#                     "start": round(cur_start, 3),
#                     "end": round(cur_end, 3),
#                 })
#                 cur_spk, cur_text = spk, [txt]
#                 cur_start, cur_end = s_start, s_end

#         if not diarize_failed and cur_text:
#             out.append({
#                 "speaker": cur_spk,
#                 "text": " ".join(cur_text).strip(),
#                 "start": round(cur_start, 3),
#                 "end": round(cur_end, 3),
#             })
#         return out

#     # --------------------------------------------------------------------- #
#     @staticmethod
#     def get_full_text(diarized: List[Dict[str, Any]]) -> str:
#         return "\n".join(f"{d['speaker']}: {d['text']}" for d in diarized)


    















