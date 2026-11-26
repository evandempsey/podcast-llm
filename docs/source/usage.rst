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
The output path may end in ``.wav`` or ``.mp3``. By default, a full render writes
audio beside the transcript using the configured output format.

To preview a chapter, edit its dialogue in the Markdown file, list the chapters,
then render the chapter by number:

.. code-block:: bash

   podcast-llm render episode.md --list-chapters
   podcast-llm render episode.md --chapter 2 --config voices.yaml

Listing chapters requires no provider credentials. A chapter render defaults to
``episode.chapter-2.wav`` in this example. Use ``--output`` to change its path.
The full render and Gradio can reuse the preview audio when the dialogue, speech
settings, and cache directory match.

Run the command again after an interruption. It reuses completed batches and
synthesizes the remaining dialogue. Editing a chapter invalidates that chapter's
audio. Changing the speech provider or its settings selects a different cache.
The cache lives under ``checkpoint_dir/audio``, which you can override with
``--cache-dir``.

Check or Retake a Chapter
-------------------------

.. code-block:: bash

   podcast-llm check episode.md --chapter 2 --config voices.yaml
   podcast-llm render episode.md --chapter 2 --retake --config voices.yaml

The check sends the chapter preview to OpenAI for transcription and flags
likely missing dialogue or changed words for review. It requires
``OPENAI_API_KEY`` and uses ``gpt-transcribe`` by default. Set ``audio_qa_model``
in YAML to override the model. Use ``--audio`` to check a different recording
of the chapter. The command exits with status 1 when it finds issues, or 2 when
it cannot complete the check.

A retake synthesizes the selected chapter again with the same speech settings.
Once it succeeds, full renders use that take and reuse the other chapters.
An interrupted retake leaves the previous take available for export.

Master a Full Export
--------------------

Use FFmpeg on your PATH and add ``--master`` to balance chapter levels before
encoding the full episode. The same audio checkpoints are used for mastered
and standard exports.

.. code-block:: bash

   podcast-llm render episode.md --master --config voices.yaml --output episode.wav
   podcast-llm render episode.md --master --config voices.yaml --output episode.mp3

Mastering targets −16 LUFS integrated and a −1 dBTP maximum true peak by
default. Override them in YAML with ``mastering_loudness_lufs`` and
``mastering_true_peak_dbtp``. The export logs measurements from the encoded
file and warns if loudness is more than 1 LU from the target. If the decoded
audio exceeds the true-peak ceiling after a retry, the export fails and any
previous output remains available. Mastered audio is encoded at 48 kHz.


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
3. Click **Check audio** to compare the preview against the script. Use **Retake
   chapter** to request fresh audio with the same dialogue and voices.
4. Click **Export episode** or **Export mastered episode** to download the
   complete audio and edited Markdown. Open **System log** for loudness and
   peak measurements from a mastered export.

The editor uses the same audio cache as the render command. A full export
reuses any chapter audio already saved for its dialogue and speech settings.
Upload a YAML config under **Settings** to choose the speech provider and voices.
The **Generate** tab also provides the existing direct-to-file workflow.
