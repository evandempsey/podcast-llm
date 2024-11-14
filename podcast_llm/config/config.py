"""
Configuration module for podcast generation.

This module provides configuration management for the podcast generation system through
the PodcastConfig class. It handles loading and validating configuration from environment
variables and YAML files, setting defaults, and providing a clean interface for accessing
configuration values throughout the application.

The configuration covers all aspects of podcast generation including:
- API credentials for various services (OpenAI, ElevenLabs, Google, etc.)
- LLM provider settings for different use cases
- Text-to-speech configuration and voice mappings
- Output format and directory management
- Rate limiting parameters
- Podcast metadata and episode structure

Example:
    config = PodcastConfig.load('config.yaml')
    print(config.podcast_name)
    print(config.tts_provider)

The module uses environment variables for sensitive values like API keys and a YAML
file for general configuration settings. It provides smart defaults while allowing
full customization of all parameters.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, List

import yaml
from dotenv import load_dotenv


@dataclass
class PodcastConfig:
    """
    Configuration class for podcast generation.

    Holds all configuration parameters needed for generating podcast content, including:
    - API keys for various services (OpenAI, ElevenLabs, etc.)
    - LLM provider settings for different use cases
    - Text-to-speech configuration
    - Output format and directory settings
    - Rate limiting parameters
    - Podcast metadata and structure

    The configuration can be loaded from environment variables and an optional YAML file
    using the load() class method.

    Attributes:
        google_api_key (str): API key for Google services
        elevenlabs_api_key (str): API key for ElevenLabs TTS
        openai_api_key (str): API key for OpenAI services
        tavily_api_key (str): API key for Tavily search
        anthropic_api_key (str): API key for Anthropic services
        fast_llm_provider (str): Provider to use for quick LLM operations
        long_context_llm_provider (str): Provider to use for long context operations
        tts_provider (str): Text-to-speech service provider
        tts_settings (Dict): Configuration settings for TTS
        output_format (str): Format for output audio files
        temp_audio_dir (str): Directory for temporary audio files
        output_dir (str): Directory for final output files
        rate_limits (Dict): Rate limiting settings for API calls
        checkpoint_dir (str): Directory for saving checkpoints
        podcast_name (str): Name of the podcast
        intro (str): Template for podcast intro
        outro (str): Template for podcast outro
        episode_structure (List): Structure template for podcast episodes
    """
    
    # API Keys
    google_api_key: str
    elevenlabs_api_key: str
    openai_api_key: str
    tavily_api_key: str
    anthropic_api_key: str

    # LLM config
    fast_llm_provider: str
    long_context_llm_provider: str
    embeddings_model: str
    
    # TTS Config 
    tts_provider: str
    tts_settings: Dict
    
    # Output Config
    output_format: str
    temp_audio_dir: str
    output_dir: str

    # Rate limits for TTS:
    rate_limits: Dict

    # Checkpointer confif
    checkpoint_dir: str

    podcast_name: str
    intro: str
    outro: str
    episode_structure: List
    
    @classmethod
    def load(cls, yaml_path: Optional[str] = None) -> 'PodcastConfig':
        """
        Load configuration from .env and optional yaml file
        
        Args:
            yaml_path: Optional path to yaml config file
            
        Returns:
            PodcastConfig: Loaded configuration object
        """
        # Load environment variables
        load_dotenv()
        
        # Required API keys from env
        required_env_vars = [
            'GOOGLE_API_KEY',
            'ELEVENLABS_API_KEY', 
            'OPENAI_API_KEY',
            'TAVILY_API_KEY',
            'ANTHROPIC_API_KEY'
        ]
        
        config_dict = {}
        for var in required_env_vars:
            value = os.getenv(var)
            if not value:
                raise ValueError(f'Missing required environment variable: {var}')
            config_dict[var.lower()] = value
            
        # Load and merge yaml config if provided
        if yaml_path:
            with open(yaml_path) as f:
                yaml_config = yaml.safe_load(f)
                config_dict.update(yaml_config)
        
        # Set defaults for optional configs
        defaults = {
            'fast_llm_provider': 'openai',
            'long_context_llm_provider': 'openai',
            'embeddings_model': 'openai',
            'tts_provider': 'google',
            'tts_settings': {
                'elevenlabs': {
                    'voice_mapping': {
                        'Interviewer': 'Chris',
                        'Interviewee': 'Charlie'
                    },
                    'model': 'eleven_multilingual_v2'
                },
                'google': {
                    'voice_mapping': {
                        'Interviewer': 'en-US-Journey-F',
                        'Interviewee': 'en-US-Journey-D'
                    },
                    'language_code': 'en-US',
                    'effects_profile_id': 'small-bluetooth-speaker-class-device'
                },
                'google_multispeaker': {
                    'voice_mapping': {
                        'Interviewer': 'R',
                        'Interviewee': 'S'
                    },
                    'language_code': 'en-US',
                    'effects_profile_id': 'small-bluetooth-speaker-class-device'
                }
            },
            'output_format': 'mp3',
            'temp_audio_dir': './.temp_audio',
            'output_dir': './output',
            'checkpoint_dir': './.checkpoints',
            'rate_limits': {
                'elevenlabs': {
                    'requests_per_minute': 20,
                    'max_retries': 10,
                    'base_delay': 2.0
                },
                'google': {
                    'requests_per_minute': 20,
                    'max_retries': 10,
                    'base_delay': 2.0
                }
            },
            'podcast_name': 'Podcast LLM',
            'intro': "Welcome to {podcast_name}. Today we've invited an expert to talk about {topic}.",
            'outro': "That's all for today. Thank you for listening to {podcast_name}. See you next time when we'll talk about whatever you want.",
            'episode_structure': [
                'Episode Introduction (with subsections)',
                'Main Discussion Topics (with subsections)',
                'Conclusion (with subsections)'
            ]
        }
        
        for key, value in defaults.items():
            if key not in config_dict:
                config_dict[key] = value
                
        return cls(**config_dict)

    @property
    def episode_structure_for_prompt(cls):
        """
        Format the episode structure as a string for use in prompts.

        Converts the episode_structure list into a newline-separated string with bullet points,
        suitable for inclusion in LLM prompts. Each section is prefixed with a hyphen for
        consistent formatting.

        Returns:
            str: Bullet-pointed string representation of the episode structure
        """
        return '\n'.join([f'- {section}' for section in cls.episode_structure])
