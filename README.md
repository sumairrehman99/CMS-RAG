# CMS Policy Copilot

CMS Policy Copilot is a Retrieval-Augmented Generation (RAG) application that enables users to ask questions about CMS policy documents using natural language. The application retrieves relevant document passages from a vector database and uses a large language model to generate grounded responses with source references.

---

## Features

- Natural language question answering
- Retrieval-Augmented Generation (RAG)
- Semantic document search
- ChromaDB vector database
- Hugging Face embeddings
- FastAPI REST API
- Dockerized deployment
- AWS ECS deployment

---

## Architecture

```text
         CMS PDF Documents
                 │
                 ▼
      Document Processing Pipeline
                 │
      Chunking & Text Cleaning
                 │
                 ▼
      Hugging Face Embeddings
                 │
                 ▼
            ChromaDB
                 │
                 ▼
        Similarity Search
                 │
                 ▼
          Retrieved Context
                 │
                 ▼
          Large Language Model
                 │
                 ▼
          Grounded Response
                 │
                 ▼
            FastAPI API
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| Backend | FastAPI |
| Vector Database | ChromaDB |
| Embeddings | Hugging Face |
| LLM | OpenAI |
| Evaluation | RAGAS |
| Deployment | AWS ECS |
| Containerization | Docker |

---

## Project Structure

```
cms-policy-copilot/
│
├── app/
├── ingest.py
├── answer.py
├── frontend.py
├── implementation/
├── chroma_db/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## How It Works

1. CMS policy documents are parsed and cleaned.
2. Documents are split into overlapping chunks.
3. Each chunk is converted into an embedding.
4. Embeddings are stored in ChromaDB.
5. User questions are embedded.
6. Similar document chunks are retrieved.
7. Retrieved context is sent to the LLM.
8. The model generates a grounded answer with supporting sources.

---

## Running Locally

Clone the repository

```bash
git clone https://github.com/yourusername/cms-policy-copilot.git

cd cms-policy-copilot
```

Run

```bash
docker compose up --build
```

API Documentation

```
http://localhost:8000/docs
```

---

## Example Questions

- Who is eligible for Medicare Part B?
- When does premium-free Part A apply?
- What information may be disclosed under the Privacy Act?
- How are inpatient hospital deductibles calculated?

---

## Evaluation

The retrieval pipeline was evaluated using **RAGAS**.

Metrics included:

- Faithfulness
- Answer grounding

Current evaluation achieved approximately **0.99 faithfulness** across the evaluation dataset.

---


## Future Improvements

- Hybrid keyword + semantic retrieval
- Streaming responses
- User authentication
- Conversation history
- Citation highlighting
- Query caching
- CI/CD with GitHub Actions
- Enhanced evaluation pipeline

---

## Lessons Learned

This project was built to learn:

- Retrieval-Augmented Generation
- Vector databases
- Semantic search
- Embedding models
- FastAPI
- Docker
- AWS ECS deployment
- RAG evaluation with RAGAS

---

## Screenshots

*(Add screenshots after completing the UI.)*
