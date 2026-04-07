# Personal Projects — Sachin M D

Portfolio of data science and ML engineering projects. Each project is a standalone repository with a Jupyter notebook, supporting scripts, and a README.

---

## Projects

| # | Project | Key Tech | Folder |
|---|---------|----------|--------|
| 1 | [RAG-Based Anomaly Explainer](#1-rag-based-anomaly-explainer) | LangChain, ChromaDB, HuggingFace, AWS SageMaker | `rag-anomaly-explainer/` |
| 2 | [Cloud-Native Anomaly Detection](#2-cloud-native-anomaly-detection) | PyTorch, MLflow, Docker, GCP Vertex AI | `cloud-anomaly-detection/` |
| 3 | [A/B Testing Framework for ML Models](#3-ab-testing-framework-for-ml-models) | SciPy, Statsmodels, Streamlit | `ab-testing-framework/` |
| 4 | [NLP Job Recommendation Engine](#4-nlp-job-recommendation-engine) | HuggingFace, FAISS, FastAPI | `nlp-job-recommender/` |

---

## 1. RAG-Based Anomaly Explainer
Automatically explains time-series anomalies using Retrieval-Augmented Generation. Anomalies are matched against a knowledge base of past incidents and root causes, then explained in plain language by an LLM. Deployed as a REST API on AWS SageMaker.

**Skills demonstrated:** LLMs, RAG, prompt engineering, vector search, anomaly detection, cloud deployment.

---

## 2. Cloud-Native Anomaly Detection
Unsupervised anomaly detection on multivariate time-series using a Transformer autoencoder (reconstruction error approach). Experiments tracked with MLflow, containerised with Docker, and deployed to GCP Vertex AI.

**Skills demonstrated:** PyTorch, Transformer architecture, MLOps, MLflow, Docker, GCP Vertex AI.

---

## 3. A/B Testing Framework for ML Models
Rigorous statistical experimentation framework for comparing anomaly detection model variants. Includes proportions Z-test, McNemar's test, Cohen's h effect size, power analysis, and threshold sweep — plus an interactive Streamlit dashboard.

**Skills demonstrated:** Hypothesis testing, A/B testing, statistical power, effect size, Streamlit.

---

## 4. NLP Job Recommendation Engine
Semantic job recommendation system. User profiles (skills, experience, preferences) are encoded with sentence transformers and matched against a FAISS-indexed job corpus via cosine similarity. Served via a FastAPI REST endpoint.

**Skills demonstrated:** NLP, semantic search, FAISS, HuggingFace transformers, FastAPI, REST API design.

---

## Contact
- LinkedIn: [linkedin.com/in/sachin-md](https://www.linkedin.com/in/sachin-md)
- Email: sachinmd18@gmail.com
