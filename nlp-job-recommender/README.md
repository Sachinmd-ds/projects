# NLP Job Recommendation Engine

Semantic job recommendation system using sentence transformers and cosine similarity. Users submit a profile (skills, experience, preferences) and receive ranked job matches — served via a FastAPI REST endpoint.

## Pipeline
```
Job Descriptions → Sentence Transformer → FAISS Index
                                               ↑
User Profile → Sentence Transformer → Query Vector → Top-K Search → Ranked Results
```

## Stack
- **HuggingFace Sentence Transformers** — `all-MiniLM-L6-v2` embeddings
- **FAISS** — fast vector similarity search
- **FastAPI** — REST API endpoint
- **Scikit-learn** — cosine similarity fallback
- **Pandas / Matplotlib** — analysis and visualisation

## Quickstart
```bash
pip install sentence-transformers faiss-cpu fastapi uvicorn pandas numpy scikit-learn matplotlib

# Explore in notebook
jupyter notebook nlp_job_recommender.ipynb

# Start API server
uvicorn serve:app --reload
# API docs at http://localhost:8000/docs
```

## API Usage
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "skills": ["Python", "NLP", "LLMs", "RAG", "Docker", "MLflow"],
    "experience": "4 years as Senior Data Scientist in audience measurement",
    "preferences": "Senior DS or GenAI engineer role, remote preferred",
    "top_k": 5
  }'
```

## Key Features
- Semantic matching beyond keyword search
- FAISS index for sub-millisecond retrieval at scale
- FastAPI with Pydantic validation and auto-generated docs
- Profile-to-job similarity heatmap visualisation
- Easily extendable to real job board data via API or scraping
