from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# -------- Text Cleaning --------
def clean_text(text):
    lines = text.split("\n")
    
    filtered = []
    for line in lines:
        line = line.strip()

        if len(line) < 40:
            continue
        if "@" in line:
            continue
        if "+91" in line:
            continue
        if "linkedin" in line.lower():
            continue

        filtered.append(line)

    return " ".join(filtered).lower()


# -------- PDF Extraction --------
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content

    if not text.strip():
        print(f"⚠️ No text extracted from {file_path}")

    return clean_text(text)


# -------- Base Matching --------
def match_score(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))


# -------- Strong Keyword Boost --------
def keyword_boost(text):
    keywords = {
        "python": 0.15,
        "machine learning": 0.18,
        "nlp": 0.15,
        "docker": 0.12,
        "kubernetes": 0.12
    }

    boost = 0
    for k, w in keywords.items():
        if k in text:
            boost += w

    return boost


# -------- Strong Generic Penalty --------
def generic_penalty(text):
    generic_words = [
        "application", "motivated", "passionate",
        "seeking", "role", "opportunity"
    ]

    penalty = 0
    for w in generic_words:
        if w in text:
            penalty += 0.05

    return penalty


# -------- Chunk-based Scoring (MAX instead of average) --------
def get_score_with_chunks(resume_text, jd):
    chunks = resume_text.split("  ")

    best_score = 0
    best_chunk = ""

    for chunk in chunks:
        if len(chunk.strip()) < 40:
            continue

        base = match_score(chunk, jd)
        boost = keyword_boost(chunk)
        penalty = generic_penalty(chunk)

        score = (0.8 * base) + boost - penalty

        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_score, best_chunk


# -------- Ranking --------
def rank_resumes(resumes, jd):
    results = []

    for r in resumes:
        score, chunk = get_score_with_chunks(r, jd)
        results.append((r, score, chunk))

    return sorted(results, key=lambda x: x[1], reverse=True)


# -------- Evaluation --------
def evaluate_ranking(ranked):
    print("\nEvaluation:\n")
    for i, (_, s, _) in enumerate(ranked):
        print(f"Rank {i+1} → Score: {s:.4f}")


# -------- Failure Analysis --------
def analyze_failure(ranked):
    print("\nFailure Analysis:\n")

    if len(ranked) >= 2:
        diff = ranked[0][1] - ranked[1][1]
        print(f"Score gap between top 2: {diff:.4f}")

        if diff < 0.05:
            print("⚠️ Weak separation → model uncertain")
        else:
            print("✅ Strong separation → confident ranking")


# -------- Main --------
if __name__ == "__main__":
    print("\nRunning Resume Ranking System...\n")

    jd = "Looking for experience in Kubernetes, Docker, and cloud deployment"

    resume_files = ["resume1.pdf", "resume2.pdf"]

    resumes = [extract_text_from_pdf(f) for f in resume_files]

    print("Extracted text lengths:", [len(r) for r in resumes])

    ranked = rank_resumes(resumes, jd)

    print("\nAll Ranked Candidates:\n")
    for r, s, _ in ranked:
        print(f"{s:.4f} → {r[:120]}...")

    top_k = 2
    print("\nTop Candidates:\n")
    for r, s, chunk in ranked[:top_k]:
        print(f"{s:.4f} → {r[:120]}...")

    print("\nTop Matching Chunks:\n")
    for _, s, chunk in ranked[:top_k]:
        print(f"\nScore: {s:.4f}")
        print(chunk[:200])

    evaluate_ranking(ranked)
    analyze_failure(ranked)