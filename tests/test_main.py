"""
Unit tests for the main.py module.
"""
import os
import json
import uuid
import pytest
from unittest.mock import patch, MagicMock, mock_open
from fastapi.testclient import TestClient
from fastapi import UploadFile


from main import (
    app,
    RuntimeSettings,
    TranscriptionEngine,
    get_gemini_model,
    transcribe_with_gemini,
    extract_audio_metadata,
    normalise_date,
    generate_report,
    create_llm_prompt
)


client = TestClient(app)


class TestEndpoints:
    """Tests for the FastAPI endpoints."""

    def test_root_endpoint(self):
        """Test the root endpoint returns the expected message."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Audio Analysis API ready."}

    def test_get_settings(self):
        """Test getting the runtime settings."""
        response = client.get("/settings")
        assert response.status_code == 200
        assert "transcription_engine" in response.json()
        assert "llm_model" in response.json()

    def test_update_settings(self):
        """Test updating the runtime settings."""
        new_settings = {
            "transcription_engine": "gemini",
            "llm_model": "gemini-1.5-flash"
        }
        response = client.post("/settings", json=new_settings)
        assert response.status_code == 200
        assert response.json()["transcription_engine"] == "gemini"
        assert response.json()["llm_model"] == "gemini-1.5-flash"

    @patch("main.os.path.getmtime", return_value=1672531200)
    @patch("main.transcribe_with_gemini")
    @patch("main.generate_report")
    @patch("main.save_transcript")
    @patch("main.extract_audio_metadata")
    @patch("builtins.open", new_callable=mock_open)
    # change to the following to fix this error
    def test_upload_audio_gemini(self, mock_file_open, mock_extract_metadata, mock_save_transcript, mock_generate_report, mock_transcribe, mock_getmtime):
    # def test_upload_audio_gemini(self, mock_file_open, mock_extract_metadata, 
    #                             mock_save_transcript, mock_generate_report, 
    #                             mock_transcribe):
        """Test uploading audio with Gemini transcription."""

        mock_extract_metadata.return_value = ("01/01/2023", 120.5)
        mock_transcribe.return_value = [
            {"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "text": "Hi there", "start": 1.5, "end": 2.5}
        ]
        mock_generate_report.return_value = {
            "feedback": "Good call",
            "keyTopics": ["greeting"],
            "emotions": ["neutral"],
            "sentiment": "positive",
            "output": "success",
            "riskWords": "none",
            "summary": "A friendly greeting",
            "rating": "80"
        }
        

        test_file = UploadFile(
            file=mock_open(read_data=b"test audio data")(),
            filename="test_audio.mp3"
        )
        

        with patch("app.main.runtime_settings", 
                  RuntimeSettings(transcription_engine=TranscriptionEngine.gemini)):
            with patch("app.main.UploadFile.read", return_value=b"test audio data"):
                response = client.post(
                    "/upload",
                    files={"file": ("test_audio.mp3", b"test audio data", "audio/mpeg")}
                )
        
        # Check the response
        assert response.status_code == 200
        assert "id" in response.json()
        assert response.json()["name"] == "test_audio.mp3"
        assert response.json()["date"] == "01/01/2023"
        assert response.json()["duration"] == "00:02:00"
        assert "transcript" in response.json()
        assert "report" in response.json()

    @patch("main.load_transcript")
    @patch("main.get_gemini_model")
    def test_chat_endpoint(self, mock_get_model, mock_load_transcript):
        """Test the chat endpoint."""
        mock_load_transcript.return_value = {
            "full_transcript_text": "SPEAKER_00: Hello\nSPEAKER_01: Hi there",
            "report_data": {
                "feedback": "Good call",
                "keyTopics": ["greeting"],
                "emotions": ["neutral"],
                "sentiment": "positive",
                "output": "success",
                "riskWords": "none",
                "summary": "A friendly greeting"
            },
            "oci_emotion": "Positive"
        }
        
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "I can help you analyze this call."
        mock_model.generate_content.return_value = mock_response
        mock_get_model.return_value = mock_model
        
        # Create a temporary file to ensure it exists
        with patch("os.path.exists", return_value=True):
            response = client.post(
                "/chat",
                json={
                    "transcript_id": "test-id",
                    "messages": [
                        {"role": "user", "content": "What was this call about?"}
                    ]
                }
            )
        
        assert response.status_code == 200
        assert "assistant_message" in response.json()
        assert response.json()["assistant_message"] == "I can help you analyze this call."


class TestHelperFunctions:
    """Tests for helper functions in main.py."""

    def test_normalise_date(self):
        """Test the normalise_date function."""
        assert normalise_date("2023-01-01") == "01/01/2023"
        assert normalise_date("2023:01:01") == "01/01/2023"
        assert normalise_date("2023/01/01") == "01/01/2023"
        assert normalise_date("invalid-date") is None

    @patch("os.path.getmtime")
    @patch("main.MutagenFile")  # instead of mutagen.File
    @patch("os.path.exists")
    def test_extract_audio_metadata(self, mock_exists, mock_mutagen_file, mock_getmtime):
        mock_exists.return_value = True
        mock_audio = MagicMock()
        mock_audio.info.length = 120.5
        mock_mutagen_file.return_value = mock_audio
        
        mock_audio.tags = {"TDRC": MagicMock()}
        mock_audio.tags["TDRC"].text = [MagicMock()]
        mock_audio.tags["TDRC"].text[0].year = 2023
        mock_audio.tags["TDRC"].text[0].month = 1
        mock_audio.tags["TDRC"].text[0].day = 1
        
        result = extract_audio_metadata("test.mp3")
        assert result[1] == 120.5
        # Test fallback to file mtime
        mock_mutagen_file.return_value = None
        mock_getmtime.return_value = 1672531200  # Jan 1, 2023 timestamp
        
        result = extract_audio_metadata("test.mp3")
        assert result[1] is None  # No duration

    def test_create_llm_prompt(self):
        """Test the create_llm_prompt function."""
        prompt = create_llm_prompt("Test transcript")
        assert "Test transcript" in prompt
        assert "You are an expert call analyst" in prompt
        assert "feedback" in prompt
        assert "keyTopics" in prompt
        assert "rating" in prompt
    @patch("main.get_gemini_model") 
    # @patch("app.main.get_gemini_model")
    def test_generate_report(self, mock_get_model):
        mock_model = MagicMock()
        mock_response = MagicMock()
        # mock_response.text = {
        #     "feedback": "Good call",
        #     "keyTopics": ["greeting"],
        #     "emotions": ["neutral"],
        #     "sentiment": "positive",
        #     "output": "success",
        #     "riskWords": "none",
        #     "summary": "A friendly greeting",
        #     "rating": "80"
        # }
        # to fix this error
        mock_response.text = json.dumps({
            "feedback": "Good call",
            "keyTopics": ["greeting"],
            "emotions": ["neutral"],
            "sentiment": "positive",
            "output": "success",
            "riskWords": "none",
            "summary": "A friendly greeting",
            "rating": "80"
        })

        mock_model.generate_content.return_value = mock_response
        mock_get_model.return_value = mock_model

        result = generate_report("SPEAKER_00: Hello\nSPEAKER_01: Hi there")
        assert result["feedback"] == "Good call"


class TestGeminiIntegration:
    """Tests for Gemini API integration."""

    @patch("os.environ")
    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_get_gemini_model(self, mock_gen_model, mock_configure, mock_environ):
        """Test the get_gemini_model function."""
        from main import get_gemini_model
        get_gemini_model.cache_clear()

        # Set up the environment
        mock_environ.get.return_value = "fake-api-key"
        
        # Call the function
        model = get_gemini_model("gemini-1.5-flash")
        
        # Check the results
        mock_configure.assert_called_once()
        mock_gen_model.assert_called_once_with("gemini-1.5-flash")

    @patch("os.path.splitext")
    @patch("builtins.open", new_callable=mock_open, read_data=b"audio data")
    @patch("google.generativeai.GenerativeModel")
    def test_transcribe_with_gemini(self, mock_gen_model, mock_file_open, mock_splitext):
        """Test the transcribe_with_gemini function."""

        mock_splitext.return_value = ("test", ".mp3")
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = """
        {
            "segments": [
                {"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 1.0},
                {"speaker": "SPEAKER_01", "text": "Hi there", "start": 1.5, "end": 2.5}
            ]
        }
        """
        mock_model.generate_content.return_value = mock_response
        mock_gen_model.return_value = mock_model
        

        result = transcribe_with_gemini("test.mp3", "gemini-1.5-flash")
        

        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[0]["text"] == "Hello"
        assert result[1]["speaker"] == "SPEAKER_01"
        assert result[1]["text"] == "Hi there"
        
        mock_response.text = "invalid json"
        result = transcribe_with_gemini("test.mp3", "gemini-1.5-flash")
        assert result[0]["speaker"] == "SYSTEM"
        assert "failed" in result[0]["text"]


if __name__ == "__main__":
    pytest.main(["-v", "test_main.py"])
