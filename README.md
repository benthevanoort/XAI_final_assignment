# XAI final assignment
In this project a BERT resume screening classifier is trained. The classifier is evaluated with two explanaibility methods: Testing with Concept Activation Vectors (TCAV) and Counterfactual. 

The goal is to analyze the influence of human-interpretable concepts on the prediction of the model. In this case the influence of technical skills, education level, leadership, gender and level of experience, on resume texts is analyzed. 

## Project Files

- `data_utils.py`  
- `train.py`
- `tcav_analysis.py`
- `counterfactuals.py`
  
## Dataset

Dataset used:

- `ranaatef/Resume-Screening-Dataset`

This project uses a subset of the big dataset to reduce runtime. These are the amount of instances used:

- Train: 500 resumes
- Validation: 125 resumes
- Test: 125 resumes

## Requirements

Install the main dependencies:

```bash
pip install torch transformers datasets accelerate scikit-learn matplotlib numpy
```

## How to run this project

Run the following scripts in this order:

### 1. Train the model

```bash
python train.py
```

This saves the trained classifier to:

```bash
./results/best_model
```

### 2. Run the TCAV analysis

```bash
python tcav_analysis.py
```

This creates:

```bash
tcav_scores.png
```

Interpretation:

- TCAV score > 0.5: the concept positively influences the prediction towards `selected`
- TCAV score around 0.5: no effect on the predicition
- TCAV score < 0.5: the concept negatively influences the prediction away from `selected`

### 3. Run the counterfactual analysis

```bash
python counterfactuals.py
```

This creates:

```bash
counterfactuals.png
```

Interpretation:

- Positive delta: swapping the keywords increases the probability of the outcome being `selected`
- Negative delta: swapping the keywords decreases the probability of the outcome being `selected`
- Delta around zero: swapping the keywords has little effect on the prediction
