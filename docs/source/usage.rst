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


Render a Saved Script
---------------------

Pass a Markdown transcript produced by ``--text-output`` to ``render``:

.. code-block:: bash

   podcast-llm render episode.md --config voices.yaml --output episode.wav

The command reads the speaker turns under ``## Script``. Chapter headings
group the turns, and trailing source references stay in the transcript.
The output path may end in ``.wav`` or ``.mp3``. If omitted, the audio file
is written beside the transcript using the configured output format.

Run the command again after an interruption. It reuses completed batches and
synthesizes the remaining dialogue. Editing a chapter invalidates that chapter's
audio. Changing the speech provider or its settings selects a different cache.
The cache lives under ``checkpoint_dir/audio``, which you can override with
``--cache-dir``.


Configuration
------------

The system can be configured using the ``config.yaml`` file:


Launching the Web Interface
-------------------------

You can launch the Gradio web interface using:

.. code-block:: bash

   podcast-llm-gui

Open a saved Markdown transcript on the **Episode editor** tab. To start from a
topic or source files, use **Draft script for editing** on the **Generate** tab.

1. Choose a chapter and edit its dialogue. Keep the speaker labels and any
   source references in the Markdown.
2. Click **Render chapter** and press **Play** to hear the edited chapter.
3. Click **Export episode** to download the complete audio and edited Markdown.

The editor uses the same audio cache as the render command. A full export
reuses any chapter audio already saved for its dialogue and speech settings.
Upload a YAML config under **Settings** to choose the speech provider and voices.
The **Generate** tab also provides the existing direct-to-file workflow.
