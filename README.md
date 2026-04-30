# AI Resume Ranking System

## Problem
Traditional keyword-based resume screening fails to capture semantic relevance and produces weak ranking.

## Solution
Built a semantic retrieval pipeline that ranks resumes using:
- chunk-based retrieval
- embedding similarity (Sentence Transformers)
- weighted keyword boosting
- penalty filtering for generic content

## Features
- PDF resume parsing
- Hybrid scoring system
- Top-K ranking
- Failure analysis & confidence detection
- Explainability via top matching chunks

## Example Output
Score gap between candidates:
0.246 vs 0.111 → strong separation

## Key Learnings
- Averaging scores reduces ranking quality
- Chunking improves retrieval signal
- Scoring design impacts performance more than model choice

## Limitations
- No structured skill extraction
- Limited handling of implicit skills