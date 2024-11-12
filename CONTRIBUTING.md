# Contributing to Podcast LLM

Thank you for your interest in contributing to Podcast LLM! This document outlines the guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/podcast-llm.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. Copy the sample .env and update with your API keys:
   ```bash
   cp podcast_llm/.env.example podcast_llm/.env
   ```

## Code Style

- Use type hints for all function parameters and return values
- Follow PEP 8 guidelines
- Use descriptive variable names
- Write docstrings for all functions and classes
- Keep functions small and focused on a single task
- Use single quotes for strings unless double quotes are needed

## Making Changes

1. Make your changes in your feature branch
2. Add tests for any new functionality
3. Ensure all tests pass: `pytest`
4. Update documentation as needed
5. Commit your changes with clear, descriptive commit messages

## Submitting Changes

1. Push to your fork: `git push origin feature/your-feature-name`
2. Create a Pull Request from your fork to the main repository
3. Describe your changes in detail in the PR description
4. Wait for review and address any feedback

## Questions?

Feel free to open an issue if you have any questions or need clarification.

Thank you for contributing to Podcast LLM!

