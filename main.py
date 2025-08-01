from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from predictor import predict_verdict, MODEL_NAME

# --- 1. Create the FastAPI app ---
app = FastAPI(
    title="Legal Verdict Prediction API",
    description="An API to predict 'Accepted' or 'Rejected' verdicts for legal case summaries using a Legal-BERT model.",
    version="1.0.0",
)

# --- 2. Define the request data model ---
class CaseRequest(BaseModel):
    case_summary: str

# --- 3. Update the API root endpoint to serve the HTML page ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # The fix is adding encoding="utf-8" to the line below
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# --- 4. Create the prediction endpoint (This stays the same) ---
@app.post("/predict/")
def create_prediction(request: CaseRequest):
    """
    Receives a case summary and returns a verdict prediction.
    - **case_summary**: The text of the legal case summary.
    """
    prediction = predict_verdict(request.case_summary)
    return prediction