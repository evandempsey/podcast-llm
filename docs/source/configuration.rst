Configuration Guide
===================

Basic Usage
-----------

Here's how to configure the Podcast LLM system by providing a custom configuration file:

.. code-block:: bash 
   python -m podcast_llm.generate "Artificial Intelligence" --config path/to-config.yml

YAML configuration
------------------

The system can be configured using the ``config.yaml`` file:

The configuration file contains several sections:

LLM Configuration
~~~~~~~~~~~~~~~
- ``fast_llm_provider``: Provider for quick LLM operations (options: 'openai', 'google', 'anthropic')
- ``long_context_llm_provider``: Provider for operations requiring longer context
- ``context_writer_model``: Gemini Developer API text model that supports
  structured output. ``gemini-4-neon`` writes both hosts in one pass by default.
- ``context_editor_model``: Optional model for a second editorial pass.

Context mode uses ``GOOGLE_API_KEY`` for writing and editing. Set the audience
and target runtime in Gradio or with ``--audience`` and ``--target-minutes``.
The target runtime is approximate. The Markdown transcript includes chapters
and references to the supplied sources. ``--qa-rounds`` applies in research mode.

Text-to-Speech Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~ 
- ``tts_provider`` selects ``google``, ``google_multispeaker``, ``elevenlabs``,
  ``elevenlabs_v4``, or ``gemini_3_8_flash_tts``.
- ``tts_settings``: Provider-specific settings including:
   - Voice mappings for interviewer and interviewee
   - Model settings
   - Language codes
   - Audio effect profiles

To use the Gemini dialogue backend, set ``GOOGLE_API_KEY`` for the Gemini API.
Change either voice to a `Gemini voice name <https://ai.google.dev/gemini-api/docs/speech-generation#voices>`_
in your YAML:

.. code-block:: yaml

   tts_provider: gemini_3_8_flash_tts
   tts_settings:
     gemini_3_8_flash_tts:
       voice_mapping:
         Interviewer: Aoede
         Interviewee: Charon

For Eleven v4 Text to Dialogue, set ``ELEVENLABS_API_KEY`` and use
`ElevenLabs voice IDs <https://elevenlabs.io/docs/eleven-creative/voices/voice-library>`_:

.. code-block:: yaml

   tts_provider: elevenlabs_v4
   tts_settings:
     elevenlabs_v4:
       voice_mapping:
         Interviewer: iP95p4xoKVk53GoZ742B  # Chris
         Interviewee: IKne3meq5aSn9XLyUdCD  # Charlie

The listed voices are the defaults for the two dialogue backends. You can
override either speaker independently. A custom Python entry point can also
call ``podcast_llm.text_to_speech.register_tts_backend(name, render)`` before
generating audio. The renderer takes the loaded configuration and the ordered
speaker/text turns and yields ``AudioChunk`` objects with audio bytes and a file
extension. Set ``tts_provider`` to the registered name.

Audio Settings
~~~~~~~~~~~~
- ``output_format``: Format for generated audio (options: 'mp3', 'wav')
- ``temp_audio_dir``: Directory for temporary audio files
- ``output_dir``: Directory for final output files

Checkpointing Settings
~~~~~~~~~~~~~~~~~~~~~~

- ``checkpoint_dir``: Directory for saving generation checkpoints

Rate Limiting
~~~~~~~~~~~
Configure API rate limits per provider:

- ``requests_per_minute``: Maximum requests allowed per minute
- ``max_retries``: Number of retry attempts
- ``base_delay``: Base delay between retries

Content Settings
~~~~~~~~~~~~~
- ``podcast_name``: Name of the podcast
- ``intro``: Template for podcast introduction (variables: {podcast_name}, {topic})
- ``outro``: Template for podcast conclusion
- ``episode_structure``: List defining the structure of episodes

Example Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # LLM Configuration
   fast_llm_provider: anthropic
   long_context_llm_provider: google

   # TTS Configuration  
   tts_provider: elevenlabs
   tts_settings:
     elevenlabs:
       voice_mapping:
         Interviewer: Chris
         Interviewee: Charlie
       model: eleven_multilingual_v2

   # Output settings
   output_format: mp3
   output_dir: ./output

   # Content settings
   podcast_name: My AI Podcast
   intro: "Welcome to {podcast_name}. Today we're exploring {topic}."
