Usage Guide
==========

Basic Usage
----------

Here's how to use the Podcast LLM system:

1. Generate a podcast about a topic:

   .. code-block:: bash

      # Research mode (default) - automatically researches the topic
      poetry run podcast-llm "Artificial Intelligence"

      # Context mode - uses provided sources
      poetry run podcast-llm "Machine Learning" --mode context --sources paper.pdf https://example.com/article

2. Options:

   .. code-block:: bash 
    
      # Customize number of Q&A rounds per section
      poetry run podcast-llm "Linux" --qa-rounds 3

      # Disable checkpointing 
      poetry run podcast-llm "Space Exploration" --checkpoint false

      # Generate audio output
      poetry run podcast-llm "Quantum Computing" --audio-output podcast.mp3

      # Generate Markdown output
      poetry run podcast-llm "Machine Learning" --text-output podcast.md

Configuration
------------

The system can be configured using the ``config.yaml`` file:
