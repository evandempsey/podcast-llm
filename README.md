# Podcast-LLM: AI-Powered Podcast Generation

An intelligent system that automatically generates engaging podcast conversations using LLMs and text-to-speech technology.

[View Documentation](https://evandempsey.github.io/podcast-llm/)

## Features

- Automated research and content gathering using Tavily search
- Dynamic podcast outline generation
- Natural conversational script writing with multiple Q&A rounds
- High-quality text-to-speech synthesis using Google Cloud or ElevenLabs
- Checkpoint system to save progress and resume generation
- Configurable voices and audio settings

## Examples

Listen to sample podcasts generated using Podcast-LLM:

### UFO Crash Retrieval (Elevenlabs voices)

[![Play Podcast Sample](https://img.shields.io/badge/Play%20Podcast-brightgreen?style=for-the-badge&logo=soundcloud)](https://soundcloud.com/evan-dempsey-153309617/ufo-crash-retrieval-elevenlabs-voices)

### Robotic Process Automation (Google voices)

[![Play Podcast Sample](https://img.shields.io/badge/Play%20Podcast-brightgreen?style=for-the-badge&logo=soundcloud)](https://soundcloud.com/evan-dempsey-153309617/robotic-process-automation-google-voices)


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/podcast-llm.git
   cd podcast-llm
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_openai_key
   GOOGLE_API_KEY=your_google_key 
   ELEVENLABS_API_KEY=your_elevenlabs_key
   TAVILY_API_KEY=your_tavily_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## Usage

1. Generate a podcast about a topic:
   ```bash
   python -m podcast_llm.generate "Artificial Intelligence"
   ```

2. Options:
   ```bash
   # Customize number of Q&A rounds per section
   python -m podcast_llm.generate "Linux" --qa-rounds 3

   # Disable checkpointing
   python -m podcast_llm.generate "Space Exploration" --checkpoint false

   # Generate audio output
   python -m podcast_llm.generate "Quantum Computing" --audio-output podcast.mp3

   # Generate Markdown output
   python -m podcast_llm.generate "Machine Learning" --text-output podcast.md
   ```

3. Customize voices and other settings in `config/config.yaml`

## License

This project is licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

This means you are free to:
- Share: Copy and redistribute the material in any medium or format
- Adapt: Remix, transform, and build upon the material

Under the following terms:
- Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made
- NonCommercial: You may not use the material for commercial purposes
- No additional restrictions: You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits

For commercial use, please contact evandempsey@gmail.com to obtain a commercial license.

The full license text can be found at: https://creativecommons.org/licenses/by-nc/4.0/legalcode

## Acknowledgements

This project was inspired by [podcastfy](https://github.com/souzatharsis/podcastfy), which provides a framework for generating podcasts using LLMs. 

This implementation differs by automating the research and content gathering process, allowing for fully autonomous podcast generation about any topic without requiring manual research or content curation.
