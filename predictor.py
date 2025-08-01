from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from lime.lime_text import LimeTextExplainer
import traceback

# --- 1. Load Model and Tokenizer ---
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.eval()
print("✅ Model and tokenizer loaded successfully.")

# --- 2. Define Labels ---
LABELS = {0: "Rejected", 1: "Accepted"}
CLASS_NAMES = ["Rejected", "Accepted"]

# --- 3. Create LIME Explainer components ---
def predictor_for_lime(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probas = torch.softmax(logits, dim=1).cpu().numpy()
    return probas

explainer = LimeTextExplainer(class_names=CLASS_NAMES)

# --- 4. Prediction Function (LIME ENABLED) ---
def predict_verdict(case_summary: str) -> dict:
    """
    Takes a legal case summary string and returns a prediction dictionary,
    including a LIME explanation.
    """
    if not case_summary:
        return {"error": "Input case summary cannot be empty."}

    try:
        # --- Standard Prediction ---
        inputs = tokenizer(case_summary, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze()
        predicted_index = torch.argmax(probabilities).item()
        predicted_label = LABELS[predicted_index]
        confidence_score = probabilities[predicted_index].item()
        
        # --- LIME Explanation is ENABLED ---
        # This is the part that generates the visual explanation.
        explanation = explainer.explain_instance(
            case_summary,
            predictor_for_lime,
            num_features=10, 
            labels=(predicted_index,)
        )

        return {
            "predicted_verdict": predicted_label,
            "confidence": f"{confidence_score:.4f}",
            "details": {
                "rejected_confidence": f"{probabilities[0].item():.4f}",
                "accepted_confidence": f"{probabilities[1].item():.4f}",
            },
            # Return the HTML explanation from LIME
            "explanation_html": explanation.as_html()
        }

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        traceback.print_exc()
        return {"error": "An internal error occurred during prediction."}