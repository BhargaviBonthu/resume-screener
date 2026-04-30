from sentence_transformers import SentenceTransformer, util

print("Loading model...")

model = SentenceTransformer('all-MiniLM-L6-v2')

print("Model loaded!")

def match_score(resume, jd):
    emb1 = model.encode(resume, convert_to_tensor=True)
    emb2 = model.encode(jd, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)
    return float(score)

if __name__ == "__main__":
    print("Running test...")

    jd = "Looking for Python ML engineer with NLP experience"
    resume = "Worked on machine learning models using python and NLP"

    score = match_score(resume, jd)
    print(f"Match Score: {score:.4f}")