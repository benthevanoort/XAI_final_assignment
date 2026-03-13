import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification

from data_utils import load_and_prepare, SAVED_MODEL_PATH

print("Loading data...")
train, val, test, train_texts_raw, test_texts_raw, tokenizer = load_and_prepare()

print("Loading model from", SAVED_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(SAVED_MODEL_PATH)

LAYER_IDX    = 8   #probe layer
N_CONCEPT    = 50  # max amount of resumes per concept
N_TEST_TCAV  = 50  # amount of test resumes per concept
TARGET_CLASS = 1   # 1 = "selected"

device = torch.device("cpu")
model.to(device)
model.eval()

CONCEPTS = {
    "technical_skills": ["python", "java", "sql", "machine learning", "javascript", "c++", "software", "programming"],
    "education":        ["bachelor", "master", "degree", "university", "phd", "graduate", "hbo", "college"],
    "leadership":       ["managed", "led", "director", "manager", "team lead", "supervised"],
    "gender_female":    ["she", "her", "woman", "ms", "mrs", "female"],
    "gender_male":      ["he", "him", "man", "mr", "mister", "male"],
    "experience_old":   ["senior", "years experience", "10 years", "principal", "veteran"],
    "experience_young": ["junior", "beginner", "just graduated"]
}

def filter_texts(texts, keywords, n): #return n resumes that contain a keyword
    return [t for t in texts if any(kw in t.lower() for kw in keywords)][:n]

def get_cls_activations(texts):
    all_acts = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i+16]
        enc = tokenizer(batch, padding="max_length", truncation=True,
                        max_length=256, return_tensors="pt").to(device)  #tokenizes batch of texts for BERT
        with torch.no_grad():
            out = model.bert(**enc, output_hidden_states=True) #runs BERT encoder without gradients
        cls = out.hidden_states[LAYER_IDX + 1][:, 0, :].cpu().numpy() #extracts [CLS] token activation from specified layer
        all_acts.append(cls)
    return np.vstack(all_acts) #returns matrix of activation vectors 

def compute_tcav_score(texts, cav):
    cav_norm = cav / (np.linalg.norm(cav) + 1e-8) #normalizes the concept direction vector (CAV), +1e-8 is added so that division by zero is not possible
    positive = 0 #counter for amount of resumes that have a positive alignment with the concept
    for text in texts:
        enc = tokenizer(text, padding="max_length", truncation=True,
                        max_length=256, return_tensors="pt").to(device) #tokenizes text for BERT
        captured = {}
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output #extracts the hidden state
            h.retain_grad()
            captured["h"] = h #saves the reference
        handle = model.bert.encoder.layer[LAYER_IDX].register_forward_hook(hook_fn) #registers a forward hook to capture the hidden state
        model.zero_grad()
        logits = model(**enc).logits #runs full model 
        logits[0, TARGET_CLASS].backward()
        handle.remove()
        if "h" in captured and captured["h"].grad is not None:
            grad = captured["h"].grad[0, 0, :].cpu().numpy()
            if np.dot(grad, cav_norm) > 0: #dot product > 0 gradient allignes with increasing the probabilty that candidate is selected, dot product < 0 
                positive += 1 
    return positive / len(texts)



random.seed(42) 
random_texts = random.sample(train_texts_raw, min(N_CONCEPT, len(train_texts_raw))) #picks random resumes from train set
random_acts  = get_cls_activations(random_texts) 
test_sample  = test_texts_raw[:N_TEST_TCAV]

tcav_scores = {}
for concept_name, keywords in CONCEPTS.items(): #loops over the concepts
    concept_texts = filter_texts(train_texts_raw, keywords, N_CONCEPT) #filters texts with keyword
    if len(concept_texts) < 7:
        print(f"Skipping '{concept_name}': too few examples ({len(concept_texts)})")
        continue

    concept_acts = get_cls_activations(concept_texts)
    n = min(len(concept_texts), len(random_acts)) #makes sure that the number of concepts and random sets are the same
    X = np.vstack([concept_acts[:n], random_acts[:n]])
    y = np.array([1] * n + [0] * n)

    #trains linear classifier to find CAV
    cav_clf = LogisticRegression(max_iter=1000)
    cav_clf.fit(X, y)
    cav = cav_clf.coef_[0]

    #computes TCAV score
    print(f"Scoring concept '{concept_name}'...")
    score = compute_tcav_score(test_sample, cav)
    tcav_scores[concept_name] = score
    print(f"  → TCAV score: {score:.3f}")


colors = ["blue", "orange", "green", "purple", "red", "pink", "yellow"]
labels = list(tcav_scores.keys())
values = list(tcav_scores.values())
positions = np.arange(len(labels)) * 1.4

plt.figure(figsize=(12, 6))
plt.bar(positions, values, width=0.8,
    color=colors[:len(tcav_scores)])
plt.xticks(positions, labels, rotation=20, ha="right")
plt.axhline(0.5, linestyle="--", color="gray", label="Chance (0.5)")
plt.ylim(0, 1)
plt.ylabel("TCAV Score")
plt.title(f"TCAV Scores — Layer {LAYER_IDX} — Class: Selected")
plt.legend()
plt.tight_layout()
plt.savefig("tcav_scores.png", dpi=150)
plt.show()
print("Saved tcav_scores.png")
