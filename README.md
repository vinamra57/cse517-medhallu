# cse517-medhallu

A project using uv for Python package management and dependency tracking.

## Prerequisites

- Python 3.8+
- `uv` package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cse517-medhallu.git
cd cse517-medhallu
```

2. Install dependencies:
```bash
uv sync
```

This installs all dependencies specified in `pyproject.toml` and locks them in `uv.lock`.

## Project Structure

```
cse517-medhallu/
├── pyproject.toml      # Project config and dependencies
├── uv.lock             # Locked dependency versions
├── .gitignore          # Git ignore rules
└── dataset_generation/   # Dataset Generation code
└── eval/   # Source code   # Evaluation code
└── similarity_experiment/   # Similarity code
└── variance_abalation/   # Variance analysis code
```

## Configuration

1. Add your OpenAI/Groq API key to a .env file:
```
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

2. For HuggingFace models, we recommend using llama_server to run models locally for faster inference:
```bash
# Install llama-server if you haven't already
pip install llama-server

# Start a HuggingFace GGUF model on a local port
llama-server -hf bartowski/Qwen2.5-3B-Instruct-GGUF:F16 --port 8087
```
Be sure to use the models and models mentioned in this table:

### Available Models
 
| Model | Port |
|-------|------|
| `Qwen/Qwen2.5-7B-Instruct` | 8085 |
| `google/gemma-2-2b-it` | 8086 |
| `Qwen/Qwen2.5-3B-Instruct` | 8087 |
| `BioMistral/BioMistral-7B` | 8088 |
| `TsinghuaC3I/Llama-3.1-8B-UltraMedical` | 8089 |
| `meta-llama/Llama-3.1-8B-Instruct` | 8090 |

 Note that it is possible that you may run out of memory when trying to run all of this. For this, make sure to only have a few models active at any time and/or reduce the quantization of the models used.

## Usage

Run each of the experiment scripts:
```bash
./generate_data.sh
./eval_models.sh
./similarity.sh
./variance.sh
```

It is estimated to take around 10-12 hours to run this code. Data is automatically downloaded if not present and all of the experiments we performed in the reproduction paper are run.
