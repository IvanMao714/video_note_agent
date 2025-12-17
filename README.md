# Video Note Agent

An intelligent agent system that automatically generates comprehensive study notes from video lectures and PDF slides using AI-powered analysis.

## Features

- **Parallel Processing**: Analyzes slides and videos concurrently for optimal performance
- **Slide Analysis**: Uses vision-capable LLMs to extract structured content from PDF slides
- **Video Transcription**: Automatic speech recognition (ASR) for video-to-text conversion
- **Intelligent Note Generation**: Creates detailed, structured study notes combining slide content and video transcripts
- **Image Matching**: Automatically matches relevant slide images to generated notes
- **Multi-LLM Support**: Supports multiple LLM providers (OpenAI, Dashscope, etc.)
- **Persistent Storage**: Optional PostgreSQL integration for state persistence and long-term storage
- **OSS Integration**: MinIO support for file storage and caching

## Architecture

The system is built using LangGraph to create a workflow that processes inputs in parallel:

```
START → [Slides Analysis] ─┐
        [Video Analysis] ────┼→ Note Generation → END
```

### Workflow Nodes

1. **Slides Analysis Node**: Analyzes PDF slides using vision LLMs, extracting:

   - Executive Summary
   - Visual Analysis
   - Textual Content
   - Key Insights
2. **Video Analysis Node**: Transcribes video content using ASR services
3. **Note Agent Node**: Generates comprehensive study notes by combining:

   - Video transcripts
   - Structured slide analyses
   - User queries (optional)

## Requirements

- Python >= 3.12
- PostgreSQL (optional, for persistent storage)
- MinIO or compatible S3 storage (optional, for file storage)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd video_note_agent
```

2. Install dependencies:

```bash
pip install -e .
```

Or using uv:

```bash
uv pip install -e .
```

## Configuration

Create a `conf.yaml` file in the project root with your configuration:

```yaml
# LLM Configuration
BASIC_MODEL:
  platform: "openrouter"  # or "alibailian", "openai", etc.
  model: "openai/gpt-4o"
  api_key: "your-api-key"

VISION_MODEL:
  platform: "alibailian"
  model: "qwen-vl-plus"
  api_key: "your-api-key"

ASR_MODEL:
  platform: "alibailian"
  model: "fun-asr"
  api_key: "your-api-key"
  oss: "minio"

# OSS Configuration (for file storage)
MINIO:
  platform: "minio"
  endpoint: "localhost:9000"
  access_key: "your-access-key"
  secret_key: "your-secret-key"
  secure: false
  bucket_name: "notes"

# PostgreSQL Configuration (optional, for persistence)
POSTGRES_MEMORY:
  host: "localhost"
  port: 5432
  database: "video_notes"
  user: "your-user"
  password: "your-password"
```

### Environment Variables

You can also configure via environment variables using the pattern:

- `{TYPE}_MODEL__{KEY}` for LLM config (e.g., `BASIC_MODEL__api_key`)
- `{TYPE}__{KEY}` for OSS config (e.g., `MINIO__endpoint`)
- `{TYPE}__{KEY}` for Memory config (e.g., `POSTGRES__host`)

## Usage

### Basic Usage

```python
from src.graph.builder import build_graph
from src.graph.state import AgentState

# Build the graph
graph = build_graph()

# Prepare input state
initial_state = {
    "slides_input_path": "path/to/slides.pdf",
    "video_input_path": "path/to/video.mp4",
    "user_query": "Generate notes focusing on key concepts",
    "messages": [],
}

# Run the graph
result = await graph.ainvoke(initial_state)

# Access results
notes = result.get("notes")
slides_list = result.get("slides_list")
video_transcript = result.get("video_transcript")
```

### Command Line Usage

```bash
python test_graph.py \
    --slides example/02-map-reduce.pdf \
    --video example/cs336_01.mp4 \
    --query "Focus on the main concepts" \
    --memory
```

### With Persistent Storage

```python
from src.memory import get_store_by_type
from src.graph.builder import build_graph

# Get PostgreSQL store
store = get_store_by_type("postgres")

# Build graph with store
graph = build_graph(store=store)

# Use thread_id for conversation tracking
config = {"configurable": {"thread_id": "conversation-1"}}
result = await graph.ainvoke(initial_state, config)
```

## Project Structure

```
video_note_agent/
├── src/
│   ├── config/          # Configuration management
│   ├── graph/           # LangGraph workflow definitions
│   │   ├── builder.py   # Graph construction
│   │   ├── nodes.py     # Workflow nodes
│   │   ├── state.py     # State definitions
│   │   └── slide_analysis/  # Slide analysis utilities
│   ├── llms/            # LLM provider integrations
│   ├── memory/          # Memory/storage providers
│   ├── oss/             # Object storage providers
│   ├── prompts/         # Prompt templates
│   └── log/             # Logging utilities
├── conf.yaml            # Configuration file
├── test_graph.py        # Example usage script
└── README.md
```

## Supported Providers

### LLM Providers

- OpenAI (via `langchain-openai`)
- Alibaba Dashscope (via custom provider)
- OpenRouter
- Azure OpenAI

### Storage Providers

- PostgreSQL (via `langgraph-checkpoint-postgres`)
- MinIO (S3-compatible)

## Development

### Running Tests

```bash
python test_graph.py --slides example/02-map-reduce.pdf --video example/cs336_01.mp4
```

### Code Style

The project follows standard Python conventions with type hints and comprehensive docstrings.

## License


## Contributing
