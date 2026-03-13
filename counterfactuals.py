import torch
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoModelForSequenceClassification

from data_utils import load_and_prepare, SAVED_MODEL_PATH

if __name__ == "__main__":
    print("Loading data...")
    train, val, test, train_texts_raw, test_texts_raw, tokenizer = load_and_prepare()

    print("Loading model from", SAVED_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(SAVED_MODEL_PATH)

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    #words that are swapped. The words are chosen based on education level, gender, technical skills and experience.
    COUNTERFACTUAL_SWAPS = [
        ("python",           "java"),
        ("bachelor",         "master"),
        ("managed",          "led"),
        ("java",             "word processing"),
        ("university",       "college"),
        ("master",           "certificate"),
        ("man",              "woman"),
        ("senior",           "junior"),
        ("she",              "he"),
        ("mr",               "ms"),
        ("years",            "months")
    ]

    N_CF = 125  # amount of test resumes

    #takes a resume and returns the probability that the outcome for the candidate is 'selected'
    def predict_proba(text):
        enc = tokenizer(text, padding="max_length", truncation=True,
                        max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[0, 1].item()

    
    cf_results = []
    for resume in test_texts_raw[:N_CF]:
        original_prob = predict_proba(resume) # hiring probability for original resume
        for keyword, replacement in COUNTERFACTUAL_SWAPS: #for every resume swap the keyword if keyword excists in resume
            if keyword.lower() in resume.lower():
                cf_resume = re.sub(re.escape(keyword), replacement, resume, flags=re.IGNORECASE)
                cf_prob   = predict_proba(cf_resume) #hiring probability for resume with changed keyword
                cf_results.append({
                    "keyword":        keyword,
                    "replacement":    replacement,
                    "original":       round(original_prob, 3),
                    "counterfactual": round(cf_prob, 3),
                    "delta":          round(cf_prob - original_prob, 3),
                })

    
    deltas_per_swap = defaultdict(list) #dictonary with the swap as key and a empty list for the deltasas value
    for r in cf_results: 
        deltas_per_swap[f"{r['keyword']} → {r['replacement']}"].append(r["delta"]) 

    avg_deltas = {k: round(np.mean(v), 3) for k, v in deltas_per_swap.items()} #avarages the deltas for every swap
 
    print("\nAverage change in P(selected) when keyword is replaced:")
    for swap, avg in sorted(avg_deltas.items(), key=lambda x: x[1]): #sorts the swaps from most negative effect to most positive
        direction = "▼" if avg < 0 else "▲" #direction of effect depends on whether the average is smaller than 0 or not
        print(f"  {direction} {swap:40s}  Δ = {avg:+.3f}") 

    plt.figure(figsize=(10, 5))
    labels     = list(avg_deltas.keys())
    values     = list(avg_deltas.values())
    bar_colors = ["red" if v < 0 else "green" for v in values]
    plt.barh(labels, values, color=bar_colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Average Δ P(selected)")
    plt.title("Counterfactual Analysis — Effect of Keyword Replacement")
    plt.tight_layout()
    plt.savefig("counterfactuals.png", dpi=150)
    plt.show()
    print("Saved counterfactuals.png")
