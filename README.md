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
├── .env.example        # Example environment variables
├── .gitignore          # Git ignore rules
└── src/                # Source code
```

## Configuration

1. Copy `.env.example` to `.env` and add your API keys:
```bash
cp .env.example .env
```

2. Add your OpenAI/Groq API key:
```
GROQ_API_KEY=your_key_here
```

## Usage

Run the main script:
```bash
uv run python main.py
```

Or directly with Python:
```bash
python main.py
```

## Development

Add new dependencies:
```bash
uv add package_name
```

Update dependencies:
```bash
uv sync
```

## Contributing

1. Create a new branch:
```bash
git checkout -b feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Description of changes"
```

3. Push to GitHub:
```bash
git push -u origin feature-name
```

4. Create a Pull Request

## License

[Add your license here]
