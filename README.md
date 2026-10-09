# Document Q&A — RAG Project

A Retrieval-Augmented Generation (RAG) application that lets you upload a PDF and ask questions about its content. Built with LangChain, ChromaDB, and Streamlit.

## Stack

- **LLM**: OpenRouter (free tier via `langchain-openrouter`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **Vector Store**: ChromaDB (on-disk for CLI, in-memory for Streamlit)
- **Retrieval**: MMR search (`k=2`, `fetch_k=5`, `lambda_mult=0.5`)
- **UI**: Streamlit chat interface

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_key_here
```

## Usage

### Streamlit app (recommended)

Upload any PDF through the browser and chat with it — no pre-processing needed.

```bash
streamlit run app.py
```

### CLI mode

First, build the vector database from a PDF:

```bash
python create_DB.py
```

Then run the command-line Q&A loop:

```bash
python main.py
```

## How it works

1. **PDF ingestion** — pages are loaded with `PyPDFLoader` and split into 1000-char chunks (100-char overlap) using `RecursiveCharacterTextSplitter`.
2. **Embedding & indexing** — chunks are embedded with MiniLM-L6-v2 and stored in a ChromaDB collection.
3. **Retrieval** — user queries are matched against the index using Maximal Marginal Relevance (MMR) to balance relevance and diversity.
4. **Generation** — retrieved passages are injected as context into a system prompt that instructs the LLM to answer only from the provided content.

The Streamlit app (`app.py`) keeps each uploaded book in a **separate in-memory** ChromaDB collection, so it never touches the on-disk `chroma_DB` used by the CLI.
