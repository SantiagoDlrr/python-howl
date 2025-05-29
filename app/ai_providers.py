"""
ai_providers.py
Unified AI provider abstraction layer supporting multiple AI services
"""

import os
import json
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import httpx
import google.generativeai as genai
from google.generativeai import GenerationConfig

logger = logging.getLogger(__name__)

class AIProvider(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    CLAUDE = "claude"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    CUSTOM = "custom"

class AIProviderInterface(ABC):
    """Abstract interface for AI providers"""
    
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
    
    @abstractmethod
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None, 
                           json_mode: bool = False, temperature: float = 0.1) -> str:
        """Generate text response"""
        pass
    
    @abstractmethod
    async def generate_chat(self, messages: List[Dict[str, str]], 
                           temperature: float = 0.1) -> str:
        """Generate chat response"""
        pass
    
    @abstractmethod
    async def transcribe_audio(self, audio_data: bytes, mime_type: str, 
                              system_prompt: Optional[str] = None) -> str:
        """Transcribe audio (if supported)"""
        pass
    
    @abstractmethod
    def supports_audio(self) -> bool:
        """Check if provider supports audio transcription"""
        pass

class GoogleProvider(AIProviderInterface):
    """Google Gemini provider implementation"""
    
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        super().__init__(api_key, model_name, base_url)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None, 
                           json_mode: bool = False, temperature: float = 0.1) -> str:
        try:
            config = GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json" if json_mode else "text/plain"
            )
            
            if system_prompt:
                model = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_prompt,
                    generation_config=config
                )
            else:
                model = genai.GenerativeModel(
                    self.model_name,
                    generation_config=config
                )
            
            response = model.generate_content(prompt)
            return response.text or ""
        
        except Exception as e:
            logger.error(f"Google provider text generation failed: {e}")
            raise
    
    async def generate_chat(self, messages: List[Dict[str, str]], 
                           temperature: float = 0.1) -> str:
        try:
            # Convert messages to Gemini format
            gemini_messages = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                gemini_messages.append({
                    "role": role,
                    "parts": [msg["content"]]
                })
            
            response = self.model.generate_content(gemini_messages)
            return response.text or ""
        
        except Exception as e:
            logger.error(f"Google provider chat generation failed: {e}")
            raise
    
    async def transcribe_audio(self, audio_data: bytes, mime_type: str, 
                              system_prompt: Optional[str] = None) -> str:
        try:
            config = GenerationConfig(
                response_mime_type="application/json"
            )
            
            if system_prompt:
                model = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_prompt,
                    generation_config=config
                )
            else:
                model = self.model
            
            audio_blob = {"inline_data": {"mime_type": mime_type, "data": audio_data}}
            prompt = ["Transcribe this audio to JSON.", audio_blob]
            
            response = model.generate_content(prompt)
            return response.text or ""
        
        except Exception as e:
            logger.error(f"Google provider audio transcription failed: {e}")
            raise
    
    def supports_audio(self) -> bool:
        return True

class OpenAIProvider(AIProviderInterface):
    """OpenAI provider implementation"""
    
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        super().__init__(api_key, model_name, base_url)
        self.base_url = base_url or "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None, 
                           json_mode: bool = False, temperature: float = 0.1) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4000
            }
            
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"] or ""
        
        except Exception as e:
            logger.error(f"OpenAI provider text generation failed: {e}")
            raise
    
    async def generate_chat(self, messages: List[Dict[str, str]], 
                           temperature: float = 0.1) -> str:
        try:
            # Convert to OpenAI format
            openai_messages = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in messages
            ]
            
            payload = {
                "model": self.model_name,
                "messages": openai_messages,
                "temperature": temperature,
                "max_tokens": 4000
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"] or ""
        
        except Exception as e:
            logger.error(f"OpenAI provider chat generation failed: {e}")
            raise
    
    async def transcribe_audio(self, audio_data: bytes, mime_type: str, 
                              system_prompt: Optional[str] = None) -> str:
        try:
            # OpenAI Whisper API doesn't support system prompts in the same way
            # We'll use the transcriptions endpoint
            files = {
                "file": ("audio", audio_data, mime_type),
                "model": ("", "whisper-1"),
                "response_format": ("", "json")
            }
            
            if system_prompt:
                files["prompt"] = ("", system_prompt)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get("text", "")
        
        except Exception as e:
            logger.error(f"OpenAI provider audio transcription failed: {e}")
            raise
    
    def supports_audio(self) -> bool:
        return True

class ClaudeProvider(AIProviderInterface):
    """Anthropic Claude provider implementation"""
    
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        super().__init__(api_key, model_name, base_url)
        self.base_url = base_url or "https://api.anthropic.com/v1"
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
    
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None, 
                           json_mode: bool = False, temperature: float = 0.1) -> str:
        try:
            payload = {
                "model": self.model_name,
                "max_tokens": 4000,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            if json_mode:
                # Claude doesn't have a direct JSON mode, but we can request JSON in the prompt
                json_instruction = "\n\nPlease respond with valid JSON only."
                payload["messages"][0]["content"] += json_instruction
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data["content"][0]["text"] or ""
        
        except Exception as e:
            logger.error(f"Claude provider text generation failed: {e}")
            raise
    
    async def generate_chat(self, messages: List[Dict[str, str]], 
                           temperature: float = 0.1) -> str:
        try:
            # Convert to Claude format
            claude_messages = []
            system_message = None
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    claude_messages.append({
                        "role": "user" if msg["role"] == "user" else "assistant",
                        "content": msg["content"]
                    })
            
            payload = {
                "model": self.model_name,
                "max_tokens": 4000,
                "temperature": temperature,
                "messages": claude_messages
            }
            
            if system_message:
                payload["system"] = system_message
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data["content"][0]["text"] or ""
        
        except Exception as e:
            logger.error(f"Claude provider chat generation failed: {e}")
            raise
    
    async def transcribe_audio(self, audio_data: bytes, mime_type: str, 
                              system_prompt: Optional[str] = None) -> str:
        # Claude doesn't support direct audio transcription
        raise NotImplementedError("Claude doesn't support audio transcription")
    
    def supports_audio(self) -> bool:
        return False

class OllamaProvider(AIProviderInterface):
    """Ollama local provider implementation"""
    
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        super().__init__(api_key, model_name, base_url)
        self.base_url = base_url or "http://localhost:11434"
    
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None, 
                           json_mode: bool = False, temperature: float = 0.1) -> str:
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            if json_mode:
                payload["format"] = "json"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        
        except Exception as e:
            logger.error(f"Ollama provider text generation failed: {e}")
            raise
    
    async def generate_chat(self, messages: List[Dict[str, str]], 
                           temperature: float = 0.1) -> str:
        try:
            # Convert to Ollama format
            ollama_messages = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in messages
            ]
            
            payload = {
                "model": self.model_name,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"] or ""
        
        except Exception as e:
            logger.error(f"Ollama provider chat generation failed: {e}")
            raise
    
    async def transcribe_audio(self, audio_data: bytes, mime_type: str, 
                              system_prompt: Optional[str] = None) -> str:
        # Ollama doesn't support direct audio transcription
        raise NotImplementedError("Ollama doesn't support audio transcription")
    
    def supports_audio(self) -> bool:
        return False

class LMStudioProvider(AIProviderInterface):
    """LM Studio local provider implementation"""
    
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        super().__init__(api_key, model_name, base_url)
        self.base_url = base_url or "http://localhost:1234/v1"
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key and api_key != "not-needed":
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None, 
                           json_mode: bool = False, temperature: float = 0.1) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4000
            }
            
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"] or ""
        
        except Exception as e:
            logger.error(f"LM Studio provider text generation failed: {e}")
            raise
    
    async def generate_chat(self, messages: List[Dict[str, str]], 
                           temperature: float = 0.1) -> str:
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4000
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"] or ""
        
        except Exception as e:
            logger.error(f"LM Studio provider chat generation failed: {e}")
            raise
    
    async def transcribe_audio(self, audio_data: bytes, mime_type: str, 
                              system_prompt: Optional[str] = None) -> str:
        # LM Studio doesn't support direct audio transcription
        raise NotImplementedError("LM Studio doesn't support audio transcription")
    
    def supports_audio(self) -> bool:
        return False

class CustomProvider(AIProviderInterface):
    """Custom API provider implementation (OpenAI-compatible)"""
    
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        super().__init__(api_key, model_name, base_url)
        if not base_url:
            raise ValueError("Custom provider requires base_url")
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None, 
                           json_mode: bool = False, temperature: float = 0.1) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4000
            }
            
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"] or ""
        
        except Exception as e:
            logger.error(f"Custom provider text generation failed: {e}")
            raise
    
    async def generate_chat(self, messages: List[Dict[str, str]], 
                           temperature: float = 0.1) -> str:
        return await self.generate_text(
            messages[-1]["content"], 
            system_prompt=messages[0]["content"] if messages[0]["role"] == "system" else None,
            temperature=temperature
        )
    
    async def transcribe_audio(self, audio_data: bytes, mime_type: str, 
                              system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError("Custom provider audio support depends on implementation")
    
    def supports_audio(self) -> bool:
        return False

class AIProviderFactory:
    """Factory for creating AI provider instances"""
    
    @staticmethod
    def create_provider(provider: AIProvider, api_key: str, model_name: str, 
                       base_url: Optional[str] = None) -> AIProviderInterface:
        """Create an AI provider instance"""
        
        if provider == AIProvider.GOOGLE:
            return GoogleProvider(api_key, model_name, base_url)
        elif provider == AIProvider.OPENAI:
            return OpenAIProvider(api_key, model_name, base_url)
        elif provider == AIProvider.CLAUDE:
            return ClaudeProvider(api_key, model_name, base_url)
        elif provider == AIProvider.OLLAMA:
            return OllamaProvider(api_key, model_name, base_url)
        elif provider == AIProvider.LM_STUDIO:
            return LMStudioProvider(api_key, model_name, base_url)
        elif provider == AIProvider.CUSTOM:
            return CustomProvider(api_key, model_name, base_url)
        else:
            raise ValueError(f"Unsupported AI provider: {provider}")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available AI providers"""
        return [provider.value for provider in AIProvider]

# Test function
async def test_provider(provider_type: str, api_key: str, model_name: str, 
                       base_url: Optional[str] = None):
    """Test an AI provider configuration"""
    try:
        provider_enum = AIProvider(provider_type)
        provider = AIProviderFactory.create_provider(
            provider_enum, api_key, model_name, base_url
        )
        
        # Test basic text generation
        response = await provider.generate_text(
            "Say hello in a friendly way",
            temperature=0.1
        )
        
        return {
            "success": True,
            "provider": provider_type,
            "model": model_name,
            "supports_audio": provider.supports_audio(),
            "test_response": response[:100] + "..." if len(response) > 100 else response
        }
    
    except Exception as e:
        return {
            "success": False,
            "provider": provider_type,
            "error": str(e)
        }



