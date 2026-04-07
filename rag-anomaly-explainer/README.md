# RAG-Based Anomaly Explainer

Automatically explains time-series anomalies using Retrieval-Augmented Generation (RAG). Detected anomalies are matched against a knowledge base of past incidents and root causes, then explained in plain language by an LLM.

## Pipeline
```
Time-Series Data → Anomaly Detection (Z-score) → Feature Extraction
→ ChromaDB Vector Store → RAG Retrieval → LLM → Natural Language Explanation
```

## Stack
- **LangChain** — RAG chain orchestration
- **ChromaDB** — vector store for incident knowledge base
- **HuggingFace Sentence Transformers** — embeddings (`all-MiniLM-L6-v2`)
- **OpenAI GPT / local LLM** — explanation generation
- **AWS SageMaker** — deployment target

## Quickstart
```bash
pip install langchain langchain-community langchain-openai chromadb \
            sentence-transformers openai pandas numpy matplotlib scikit-learn
export OPENAI_API_KEY=your-key-here
jupyter notebook rag_anomaly_explainer.ipynb
```

## Key Features
- Rolling Z-score anomaly detection on non-stationary time-series
- Semantic retrieval of similar past incidents from vector store
- Structured LLM prompt returning root cause, confidence, and recommended action
- JSON export of all explanations
- AWS SageMaker deployment reference included

## Results
Reduces anomaly triage time by ~50% by providing automated first-pass explanations for on-call engineers.
