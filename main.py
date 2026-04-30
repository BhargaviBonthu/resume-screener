from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def match_score(resume, jd):
    emb1 = model.encode(resume, convert_to_tensor=True)
    emb2 = model.encode(jd, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)
    return float(score)

def rank_resumes(resumes, jd):
    scores = []
    for r in resumes:
        score = match_score(r, jd)
        scores.append((r, score))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    jd = "Looking for Python ML engineer with NLP experience"

    resumes = [
        "Worked on machine learning and NLP using python",
        "Frontend developer with React experience",
        "Data scientist with deep learning and NLP expertise"
    ]

    ranked = rank_resumes(resumes, jd)

    print("\nRanked Candidates:\n")
    for r, s in ranked:
        print(f"{s:.4f} → {r}")