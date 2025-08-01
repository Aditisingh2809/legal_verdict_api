import pandas as pd
import requests
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg') # <-- Add this line
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/predict/"
CSV_FILE_PATH = "evaluation_dataset.csv"  # The name of your CSV file
TEXT_COLUMN = "text"  # The name of the column with the case summaries
LABEL_COLUMN = "label"  # The name of the column with the true verdicts

def get_prediction(summary: str):
    """Calls the running API to get a prediction."""
    payload = {"case_summary": summary}
    try:
        response = requests.post(API_URL, json=payload, timeout=90) # 90-second timeout for LIME
        if response.status_code == 200:
            return response.json()["predicted_verdict"]
        else:
            print(f"API Error: Status Code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def evaluate():
    """Main function to run the evaluation."""
    print(f"Reading dataset from {CSV_FILE_PATH}...")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE_PATH}' was not found. Make sure it's in the same folder.")
        return

    predictions = []
    true_labels = list(df[LABEL_COLUMN])
    
    print("Getting predictions from the API...")
    # Use tqdm for a nice progress bar
    for text in tqdm(df[TEXT_COLUMN], desc="Evaluating"):
        prediction = get_prediction(text)
        predictions.append(prediction)

    # --- Calculate Metrics ---
    # Filter out any failed predictions
    valid_predictions = [p for p in predictions if p is not None]
    valid_true_labels = [true_labels[i] for i, p in enumerate(predictions) if p is not None]

    if not valid_true_labels:
        print("Could not get any valid predictions from the API. Please check if the server is running.")
        return

    accuracy = accuracy_score(valid_true_labels, valid_predictions)
    precision = precision_score(valid_true_labels, valid_predictions, pos_label="Accepted")
    recall = recall_score(valid_true_labels, valid_predictions, pos_label="Accepted")
    f1 = f1_score(valid_true_labels, valid_predictions, pos_label="Accepted")

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (for 'Accepted'): {precision:.4f}")
    print(f"Recall (for 'Accepted'): {recall:.4f}")
    print(f"F1-Score (for 'Accepted'): {f1:.4f}")
    print("------------------------\n")

    # --- Generate Confusion Matrix ---
    print("Generating confusion matrix...")
    cm = confusion_matrix(valid_true_labels, valid_predictions, labels=["Accepted", "Rejected"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Accepted", "Rejected"], 
                yticklabels=["Accepted", "Rejected"])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot to a file
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    evaluate()