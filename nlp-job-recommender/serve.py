"""
NLP Job Recommender — FastAPI Endpoint
Run: uvicorn serve:app --reload
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(
    title="NLP Job Recommendation API",
    description="Semantic job recommendations using sentence transformers.",
    version="1.0.0"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# ── Sample Job Data (replace with DB or file in production) ──
JOB_LISTINGS = [
    {"id": "JOB001", "title": "Senior Data Scientist", "company": "TechCorp", "location": "Bangalore",
     "description": "Build and deploy ML models for time-series forecasting and anomaly detection. Python, PySpark, Airflow, AWS, Azure."},
    {"id": "JOB002", "title": "ML Engineer", "company": "DataAI", "location": "Hyderabad",
     "description": "Operationalise ML pipelines. MLOps, Docker, Kubernetes, PyTorch, TensorFlow."},
    {"id": "JOB003", "title": "NLP Research Scientist", "company": "AI Labs", "location": "Remote",
     "description": "LLMs, RAG systems, LangChain, vector databases, transformers, production NLP."},
    {"id": "JOB004", "title": "Data Engineer", "company": "StreamBase", "location": "Pune",
     "description": "Apache Spark, Kafka, Databricks, BigQuery, Snowflake, Python, SQL."},
    {"id": "JOB005", "title": "MLOps Engineer", "company": "CloudScale", "location": "Remote",
     "description": "MLflow, Docker, Kubernetes, AWS SageMaker, GCP Vertex AI, CI/CD for ML."},
    {"id": "JOB006", "title": "GenAI Engineer", "company": "LLMCo", "location": "Remote",
     "description": "LLMs, RAG pipelines, prompt engineering, LangChain, ChromaDB, OpenAI API."},
]

df_jobs = pd.DataFrame(JOB_LISTINGS)
df_jobs["full_text"] = df_jobs["title"] + ". " + df_jobs["description"]
job_embeddings = model.encode(df_jobs["full_text"].tolist())

# ── Schemas ───────────────────────────────────────────────────
class ProfileRequest(BaseModel):
    skills: List[str] = Field(..., example=["Python", "NLP", "LLMs", "RAG", "Docker"])
    experience: str   = Field(..., example="4 years in ML and data science at Nielsen")
    preferences: str  = Field(..., example="Senior DS or GenAI role, remote preferred")
    top_k: int        = Field(default=5, ge=1, le=10)

class JobMatch(BaseModel):
    id: str
    title: str
    company: str
    location: str
    similarity_score: float

class RecommendResponse(BaseModel):
    profile_used: str
    recommendations: List[JobMatch]

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: ProfileRequest):
    profile_text = (
        f"Skills: {', '.join(request.skills)}. "
        f"Experience: {request.experience}. "
        f"Preferences: {request.preferences}."
    )
    query_vec = model.encode([profile_text])
    sims      = cosine_similarity(query_vec, job_embeddings)[0]
    top_idx   = np.argsort(sims)[::-1][:request.top_k]

    results = []
    for idx in top_idx:
        row = df_jobs.iloc[idx]
        results.append(JobMatch(
            id=row["id"], title=row["title"],
            company=row["company"], location=row["location"],
            similarity_score=round(float(sims[idx]), 4)
        ))

    return RecommendResponse(profile_used=profile_text, recommendations=results)

@app.get("/jobs")
def list_jobs():
    return df_jobs[["id", "title", "company", "location"]].to_dict(orient="records")
