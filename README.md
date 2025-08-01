# Real-Time Legal Verdict Prediction API with LIME Explainability

This project is a real-time API built with Python and FastAPI that uses a pre-trained Legal-BERT model to predict verdicts ("Accepted" or "Rejected") from legal case summaries. A key feature is the integration of LIME (Local Interpretable Model-agnostic Explanations) to provide transparency and explain why the model made a specific prediction.

This project was developed for the research paper: "Explainable AI in Legal Tech: A Real-Time Verdict Prediction API using Legal-BERT and LIME."

## 🚀 Features
- **Real-Time Prediction:** A low-latency API endpoint for instantaneous predictions.
- **Explainable AI (XAI):** Uses LIME to highlight the words that most influenced the model's decision, enhancing trust and transparency.
- **Domain-Specific Model:** Leverages Legal-BERT, a transformer model pre-trained on legal text for superior contextual understanding.
- **Web Interface:** A simple, clean UI built with HTML and JavaScript for easy demonstration and interaction.

## 🛠️ Tech Stack
- **Backend:** Python, FastAPI, Uvicorn
- **ML/NLP:** PyTorch, Hugging Face Transformers, LIME
- **Evaluation:** Scikit-learn, Pandas, Matplotlib

## ⚙️ How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create venv
    python -m venv venv
    # Activate on Windows
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    *(You must first create the `requirements.txt` file by running `pip freeze > requirements.txt` in your activated venv).*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the API server:**
    ```bash
    uvicorn main:app --reload
    ```

5.  Open your browser and navigate to `http://127.0.0.1:8000/`.
