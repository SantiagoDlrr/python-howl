"""
Unit tests for the temp_storage.py module.
"""
import os
import json
import pytest
from unittest.mock import patch, mock_open, MagicMock

# Import the functions from temp_storage.py
from python_howl.app.temp_storage import (
    save_transcript,
    load_transcript,
    STORAGE_DIR
)


class TestTempStorage:
    """Tests for the temp_storage module."""

    @patch("os.path.join")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_transcript(self, mock_json_dump, mock_file_open, mock_path_join):
        """Test saving a transcript."""
        # Set up the mocks
        mock_path_join.return_value = f"{STORAGE_DIR}/transcript_test-id.json"
        
        # Test data
        transcript_id = "test-id"
        diarized_transcript = [
            {"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "text": "Hi there", "start": 1.5, "end": 2.5}
        ]
        full_transcript_text = "SPEAKER_00: Hello\nSPEAKER_01: Hi there"
        report_data = {
            "feedback": "Good call",
            "keyTopics": ["greeting"],
            "emotions": ["neutral"],
            "sentiment": "positive",
            "output": "success",
            "riskWords": "none",
            "summary": "A friendly greeting",
            "rating": "80"
        }
        oci_emotion = "Positive"
        oci_aspects = []
        date = "01/01/2023"
        duration = "00:02:00"
        
        # Call the function
        save_transcript(
            transcript_id=transcript_id,
            diarized_transcript=diarized_transcript,
            full_transcript_text=full_transcript_text,
            report_data=report_data,
            oci_emotion=oci_emotion,
            oci_aspects=oci_aspects,
            date=date,
            duration=duration
        )
        
        # Check the results
        mock_file_open.assert_called_once_with(
            f"{STORAGE_DIR}/transcript_test-id.json", "w", encoding="utf-8"
        )
        mock_json_dump.assert_called_once()
        
        # Check the data being saved
        args, kwargs = mock_json_dump.call_args
        data_saved = args[0]
        assert data_saved["id"] == transcript_id
        assert data_saved["diarized_transcript"] == diarized_transcript
        assert data_saved["full_transcript_text"] == full_transcript_text
        assert data_saved["report_data"] == report_data
        assert data_saved["oci_emotion"] == oci_emotion
        assert data_saved["oci_aspects"] == oci_aspects
        assert data_saved["date"] == date
        assert data_saved["duration"] == duration
        
        # Check the JSON dump parameters
        assert kwargs["ensure_ascii"] is False
        assert kwargs["indent"] == 2

    @patch("os.path.join")
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    def test_load_transcript(self, mock_json_load, mock_file_open, mock_path_exists, mock_path_join):
        """Test loading a transcript."""
        # Set up the mocks
        mock_path_join.return_value = f"{STORAGE_DIR}/transcript_test-id.json"
        mock_path_exists.return_value = True
        
        # Test data
        expected_data = {
            "id": "test-id",
            "diarized_transcript": [
                {"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 1.0},
                {"speaker": "SPEAKER_01", "text": "Hi there", "start": 1.5, "end": 2.5}
            ],
            "full_transcript_text": "SPEAKER_00: Hello\nSPEAKER_01: Hi there",
            "report_data": {
                "feedback": "Good call",
                "keyTopics": ["greeting"],
                "emotions": ["neutral"],
                "sentiment": "positive",
                "output": "success",
                "riskWords": "none",
                "summary": "A friendly greeting",
                "rating": "80"
            },
            "oci_emotion": "Positive",
            "oci_aspects": [],
            "date": "01/01/2023",
            "duration": "00:02:00"
        }
        mock_json_load.return_value = expected_data
        
        # Call the function
        result = load_transcript("test-id")
        
        # Check the results
        mock_file_open.assert_called_once_with(
            f"{STORAGE_DIR}/transcript_test-id.json", "r", encoding="utf-8"
        )
        mock_json_load.assert_called_once()
        assert result == expected_data

    @patch("os.path.join")
    @patch("os.path.exists")
    def test_load_transcript_not_found(self, mock_path_exists, mock_path_join):
        """Test loading a transcript that doesn't exist."""
        # Set up the mocks
        mock_path_join.return_value = f"{STORAGE_DIR}/transcript_test-id.json"
        mock_path_exists.return_value = False
        
        # Call the function and check for exception
        with pytest.raises(FileNotFoundError):
            load_transcript("test-id")

    @patch("os.path.join")
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    def test_load_transcript_invalid_json(self, mock_json_load, mock_file_open, 
                                         mock_path_exists, mock_path_join):
        """Test loading a transcript with invalid JSON."""
        # Set up the mocks
        mock_path_join.return_value = f"{STORAGE_DIR}/transcript_test-id.json"
        mock_path_exists.return_value = True
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        # Call the function and check for exception
        with pytest.raises(ValueError):
            load_transcript("test-id")

    @patch("os.path.join")
    @patch("os.path.exists")
    @patch("builtins.open")
    def test_load_transcript_io_error(self, mock_file_open, mock_path_exists, mock_path_join):
        """Test loading a transcript with IO error."""
        # Set up the mocks
        mock_path_join.return_value = f"{STORAGE_DIR}/transcript_test-id.json"
        mock_path_exists.return_value = True
        mock_file_open.side_effect = IOError("Permission denied")
        
        # Call the function and check for exception
        with pytest.raises(IOError):
            load_transcript("test-id")


if __name__ == "__main__":
    pytest.main(["-v", "test_temp_storage.py"])
