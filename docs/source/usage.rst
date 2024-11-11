Usage Guide
==========

Basic Usage
----------

Here's how to use the Podcast LLM system:

1. Generate a podcast about a topic:

   .. code-block:: bash

      # Research mode (default) - automatically researches the topic
      python -m podcast_llm.generate "Artificial Intelligence"

      # Context mode - uses provided sources
      python -m podcast_llm.generate "Machine Learning" --mode context --sources paper.pdf https://example.com/article

2. Options:

   .. code-block:: bash 
    
      # Customize number of Q&A rounds per section
      python -m podcast_llm.generate "Linux" --qa-rounds 3

      # Disable checkpointing 
      python -m podcast_llm.generate "Space Exploration" --checkpoint false

      # Generate audio output
      python -m podcast_llm.generate "Quantum Computing" --audio-output podcast.mp3

      # Generate Markdown output
      python -m podcast_llm.generate "Machine Learning" --text-output podcast.md

Configuration
------------

The system can be configured using the ``config.yaml`` file:
