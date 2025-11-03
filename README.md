# Basic RAG - Production

RAG (Retrieval-Augmented Generation) system for querying documentation using semantic search

## Setup

### Quick Setup (Automated)

```bash
./setup.sh
```

This will automatically create venv and install dependencies.

### Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file from template
cp .env.example .env

# 5. Edit .env and add your API key
Open .env and set OPENROUTER_API_KEY=your-actual-key-here
```

Get API key at: https://openrouter.ai/keys

## Usage

Make sure that you have **knowledge** folder in the root of the project

```bash
# Activate virtual environment
source venv/bin/activate

# Run query
python script.py -m "qwen/qwen3-30b-a3b:free" -t "What technologies are used in mobile development?" -k 5
```

### Examples

```bash
# First time: Build embeddings cache (~17 minutes)
python script.py --rebuild

# Query with specific model
python script.py -m "qwen/qwen-2.5-coder-32b-instruct" -t "What is penetration testing?"

# Query with different model
python script.py -m "meta-llama/llama-3.1-8b-instruct" -t "How to build a mobile app?"

# Query with default model
python script.py -t "What is machine learning?"

# Retrieve more chunks for context
python script.py -t "Your question" -k 15
```

See all models: https://openrouter.ai/models

## Rebuild Cache

After adding new documents or updating existing ones:

```bash
python script.py --rebuild
```

Or delete cache manually:
```bash
rm -rf .cache
```

## Project Structure

```
basic_rag/
├── knowledge/              # Documentation files (md, html, erb)
├── venv/                   # Virtual environment (created by setup)
├── .cache/                 # Embeddings cache (created on first run)
├── script.py              # Main entry point (production RAG)
├── document_loader.py     # Document loading
├── text_processor.py      # Text chunking
├── vector_store.py        # Embeddings and search
├── llm_client.py          # LLM integration
├── requirements.txt       # Dependencies
├── .env                   # Your API key (create from .env.example)
├── .env.example          # API key template
├── .gitignore            # Git ignore rules
├── setup.sh              # Automated setup script
└── README.md             # This file
```

## How It Works

**First Run (Initial Setup):**
- Loads all docs from the root folder knowledge/ (markdown, HTML, ERB)
- Splits into chunks (~1000+ chunks)
- Generates embeddings using sentence-transformers
- Saves to `.cache/`

**Subsequent Runs:**
- Loads from cache instantly
- Searches using vector similarity
- Returns accurate, context-aware results
