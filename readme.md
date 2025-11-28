# Milk Production Fact Verification System

This tool is a Retrieval-Augmented Generation (RAG) system designed to verify statistical claims about milk production against an official PDF document. It uses semantic search (FAISS) to find relevant evidence and an LLM (via Groq) to determine the truthfulness of specific claims.

##  Features

* **PDF Parsing:** Extracts clean, readable text lines from PDF reports using `pdfplumber`.
* **Semantic Search:** Uses `SentenceTransformer` and `FAISS` to index document facts and retrieve context based on meaning, not just keywords.
* **AI Verification:** Sends the claim and retrieved evidence to Groq for a final "True", "False", or "Unverifiable" verdict.
* **JSON Output:** Returns structured data including the verdict, reasoning, and specific evidence found in the document.

##  Prerequisites

* Python 3.8 or higher
* A **Groq API Key** (Get one at [console.groq.com](https://console.groq.com))
* A source PDF file (default is `milk.pdf`)

##  Installation

1.  **Create a Virtual Environment** (recommended):
    ```bash
    python -m venv ok
    # Windows
    ok\Scripts\activate
    ```

2.  **Install Dependencies**:
    Create a `requirements.txt` file with the contents below, or install manually.
    
    ```bash
    pip install pdfplumber sentence-transformers faiss-cpu numpy groq python-dotenv
    ```

## Configuration

1.  **Environment Variables**:
    Create a file named `.env` in the root directory of the project. Add your Groq API key:
    
    ```env
    GROQ_API_KEY=gsk_your_actual_api_key_here
    ```

2.  **Source Data**:
    Place your source document named `milk.pdf` in the same directory as the script.

## Usage

Run the script using Python:

python hi.py