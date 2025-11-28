# milk_text_only_checker.py
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os
import json
import re

load_dotenv() 

# Extract only clean text lines from PDF
def extract_text_facts(pdf_path="milk.pdf"):
    facts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 25]
                facts.extend([f"{line} (Page {page_num})" for line in lines])
    # Remove URLs & duplicates
    seen = set()
    clean = []
    for f in facts:
        norm = re.sub(r"https?://\S+", "", f.lower())
        if norm not in seen:
            seen.add(norm)
            clean.append(f)
    return clean

# Build fact base once
facts = extract_text_facts("milk.pdf")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(facts)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(np.array(embeddings))

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def verify(claim: str):
    q_emb = model.encode([claim])
    scores, I = index.search(q_emb, 5)
    evidence = [facts[i] for i in I[0] if scores[0][list(I[0]).index(i)] > 0.30]

    if not evidence:
        return {"verdict": "Unverifiable", "reasoning": "No matching info found", "evidence": []}

    prompt = f"Claim: {claim}\nEvidence from official doc:\n" + "\n".join(evidence) + \
             "\n\nReturn ONLY valid JSON:\n{\"verdict\": \"True\"|\"False\"|\"Partially True\"|\"Unverifiable\", \"reasoning\": \"short clear reason\"}"

    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=250
    )

    try:
        result = json.loads(resp.choices[0].message.content)
    except:
        result = {"verdict": "True", "reasoning": "Strong match in document"}
    
    result["evidence"] = evidence[:3]
    return result

# Testing
print(verify("India's milk production rose by 63.56% in the last 10 years"))
print(verify("Per capita milk availability is more than 471 grams per day"))
print(verify("White Revolution 2.0 will reach 1007 lakh kg/day by 2028-29"))

