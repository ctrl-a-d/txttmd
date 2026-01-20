# txttmd

> This was vibecoded. Don't expect too much. Or anything at all.

Transform raw text notes into organized, structured markdown using LLMs.

## Overview

txttmd monitors your notes inbox and automatically processes raw text files into properly formatted markdown. It uses LLM providers to analyze content, extract metadata, categorize notes, and format them according to your preferences.

## Features

- **Auto-processing**: Monitors inbox folder and processes new notes automatically
- **Multi-provider support**: Claude, OpenAI, Groq, Ollama, OpenRouter, Perplexity
- **Smart routing**: Route notes to different LLMs based on content (length, code blocks, etc.)
- **Category detection**: Automatically categorizes notes based on content analysis
- **Flexible configuration**: YAML-based config with per-provider settings
- **Docker support**: Optional containerized deployment
- **Privacy-aware**: Choose between cloud APIs or local processing with Ollama

## Quick Start

### Prerequisites

- Python 3.10+
- API keys for your chosen provider(s), OR
- Ollama installed locally (for local-only processing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ctrl-a-d/txttmd.git
cd txttmd
```

2. Run the setup script:
```bash
./setup.sh
```

This will:
- Check Python version (3.10+ required)
- Create a virtual environment
- Install dependencies
- Launch the setup wizard

The wizard will guide you through:
- Notes path configuration
- Privacy considerations
- Provider selection
- API key setup
- Category customization
- Optional Docker deployment

### Running

**Locally:**
```bash
python -m src.main
```

**With Docker:**
```bash
cd docker
docker-compose up -d
```

## Privacy Notice

**External providers** (Claude, OpenAI, Groq, etc.):
- Complete note content is sent to their APIs
- Subject to provider data retention policies
- Not recommended for sensitive/confidential notes

**Ollama (Local)**:
- Requires separate installation from ollama.com
- Notes processed entirely on your machine
- Recommended for sensitive data

## Configuration

Configuration is stored in `config/config.yaml`:

```yaml
notes_path: /path/to/notes

folders:
  inbox: _Inbox
  output: Notes
  archive: _Archive

llm:
  providers:
    ollama:
      model: llama3.2
      enabled: true
    groq:
      model: llama-3.3-70b-versatile
      enabled: true
      api_key_env: GROQ_API_KEY

  routing:
    - provider: groq
      priority: 100
      conditions:
        - type: word_count
          value: 500
          operator: "<"
    - provider: ollama
      priority: 0
      conditions:
        - type: always

categories:
  - name: Projects
    path: Projects
    keywords: [project, task, todo]
  - name: Ideas
    path: Ideas
    keywords: [idea, brainstorm, concept]
```

## How It Works

1. Drop a `.txt` or `.md` file into your `_Inbox` folder
2. txttmd detects the new file and reads its content
3. The LLM analyzes and processes the note:
   - Extracts title and metadata
   - Determines appropriate category
   - Formats content as proper markdown
4. Processed note saved to `Notes/{Category}/`
5. Original moved to `_Archive/`

## Project Structure

```
txttmd/
├── src/
│   ├── main.py              # Entry point
│   ├── config.py            # Config loader
│   ├── file_monitor.py      # Inbox watcher
│   ├── file_handler.py      # File operations
│   ├── note_processor.py    # LLM processing
│   └── setup/
│       └── wizard.py        # Interactive setup
├── config/
│   └── config.yaml          # Main configuration
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

## Requirements

See `requirements.txt` for full list. Key dependencies:
- `watchdog` - File system monitoring
- `anthropic` - Claude API
- `openai` - OpenAI/compatible APIs
- `pyyaml` - Configuration
- `questionary` - Setup wizard
- `rich` - Terminal UI

## License

MIT License - See LICENSE file for details
