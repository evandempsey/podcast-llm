Usage Guide
==========

Basic Usage
----------

Here's how to use the Podcast LLM system:

1. Generate a podcast about a topic:

   .. code-block:: bash

      # Research mode (default) - automatically researches the topic
      podcast-llm "Artificial Intelligence"

      # Context mode - uses provided sources
      podcast-llm "Machine Learning" --mode context --sources paper.pdf https://example.com/article

2. Options:

   .. code-block:: bash 
    
      # Customize number of Q&A rounds per section
      podcast-llm "Linux" --qa-rounds 3

      # Disable checkpointing 
      podcast-llm "Space Exploration" --checkpoint false

      # Generate audio output
      podcast-llm "Quantum Computing" --audio-output podcast.mp3

      # Generate Markdown output
      podcast-llm "Machine Learning" --text-output podcast.md


Configuration
------------

The system can be configured using the ``config.yaml`` file:


Launching the Web Interface
-------------------------

You can launch the Gradio web interface using:

.. code-block:: bash

   podcast-llm-gui

This launches a user-friendly web interface where you can:

- Enter a podcast topic
- Choose between research and context modes  
- Upload source files and URLs for context mode
- Configure Q&A rounds and checkpointing
- Specify output paths for text and audio
- Monitor generation progress in real-time
