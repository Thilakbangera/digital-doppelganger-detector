from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import shap
import torch
import numpy as np

# Load model and tokenizer
model_name = "roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# SHAP-compatible prediction function
def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]  # ensure list of str
    inputs = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
    return probs

# SHAP explainer with tokenizer workaround
explainer = shap.Explainer(
    predict_proba,
    masker=shap.maskers.Text(tokenizer)
)

# Extract important phrases
def explain_text(text):
    shap_values = explainer([text])
    tokens = shap_values.data[0]
    scores = shap_values.values[0]

    explanation = []
    for token, score in zip(tokens, scores):
        scalar = score if isinstance(score, (float, int)) else score[0]
        if abs(scalar) > 0.1:
            explanation.append(token)
    return explanation

# Final detect & explain wrapper
def detect_and_explain(text):
    result = classifier(text)[0]
    label = result['label']
    confidence = result['score']
    highlighted_phrases = explain_text(text)
    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "highlighted_phrases": highlighted_phrases
    }
