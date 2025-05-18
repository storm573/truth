# Truth-Seeking MCP Server

## Overview

The Truth-Seeking MCP Server is a specialized Model Context Protocol (MCP) server designed to provide balanced perspectives on claims and statements. It helps analyze claims by finding supporting and opposing viewpoints from various sources, extracting claims from text, and transcribing media content.

## Features

- **Claim Analysis**: Analyzes claims by finding supporting and opposing perspectives from web searches, presenting balanced viewpoints with confidence scoring.
- **Claim Extraction**: Extracts factual, predictive, or normative claims from text using LLM processing.
- **Perspective History**: Retrieves historical perspectives for similar claims from memory storage for consistent analysis over time.
- **Media Transcription**: Transcribes speech from YouTube videos or podcasts into text using OpenAI's Whisper API.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/storm573/truth.git
   cd truth
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables in a `.env` file:
   ```
   BRAVE_SEARCH_API_KEY=your_brave_search_api_key
   PERPLEXITY_API_KEY=your_perplexity_api_key  # Optional but recommended
   OPENAI_API_KEY=your_openai_api_key  # Required for transcription
   HOST=0.0.0.0  # Default host
   PORT=8050  # Default port
   ENABLE_MEMORY=false  # Set to true to enable memory features
   ```

## Usage

Start the server:

```bash
python truth.py
```

The server will be available at `http://localhost:8050` (or the host/port specified in your environment variables).

## API Tools

1. **analyze_claim**: Analyze a claim by finding perspectives both supporting and opposing it.
2. **extract_claim**: Extract factual, predictive, or normative claims from text.
3. **get_perspective_history**: Get historical perspectives for similar claims.
4. **transcribe_media**: Transcribe speech from YouTube videos or podcasts into text.

## Project Information

Google Cloud Project ID: truth-seeker-460200

## License

This project is proprietary and confidential.

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.
