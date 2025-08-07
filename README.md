# 🧠 HackRx PDF QA System

A RAG-based Question Answering system for extracting answers from policy PDFs using:

* 🧾 pdfplumber – for PDF text extraction
* 🧠 SentenceTransformers + Chroma – for chunk embedding & storage
* 💬 Groq LLaMA-4 – for generating answers
* ⚡ FastAPI – to expose the pipeline as a REST API

### 🚀 Quick Start

1. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

2. Add .env
   ```GROQ_API_KEY=your_groq_api_key```

3. Run API
   ```bash
   uvicorn api:app --reload
   ```

### 📡 API Usage

`POST` /hackrx/run

Request

```json
{
  "documents": "https://example.com/sample.pdf",
  "questions": [
    "What is the grace period?",
    "Does it cover maternity?"
  ]
}
   ```

Add header:
Authorization: Bearer <token>

Response

```json
{
  "answers": [
    "30 days grace period",
    "Yes, after 9 months"
  ]
}
```

Set `rebuild = True` to reprocess PDF.

### 🧩 Tech Stack

* LangChain + ChromaDB
* SentenceTransformers
* Groq (LLaMA-4 Scout)
* FastAPI