# AI Resume Screener (Semantic Ranking System)

This project demonstrates a semantic retrieval system that ranks resumes against a job description using embedding-based similarity.

## 🚀 Features
- Semantic similarity using Sentence Transformers
- Multi-resume ranking system
- Context-aware matching (not keyword-based)

## 🧠 How it works
1. Convert resumes and job description into embeddings
2. Compute cosine similarity
3. Rank candidates based on relevance

## 📊 Example Output
0.7264 → ML + NLP profile  
0.5459 → Data science profile  
0.1706 → Unrelated frontend profile  

## 🛠 Tech Stack
- Python
- Sentence Transformers
- NLP embeddings

## 📌 Future Improvements
- PDF resume parsing
- Top-K filtering
- Web UI (Streamlit)
- Integration with vector databases