from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Define the model name we want to use from the Hugging Face Hub
model_name = "nlpaueb/legal-bert-base-uncased"
print(f"Attempting to download model and tokenizer for: {model_name}")

try:
    # Download the tokenizer for this model. The tokenizer prepares the text for the model.
    # The tokenizer will be downloaded and saved in a cache folder for future use.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Download the model. We specify the number of labels we want for our classification task.
    # For our project ("Accepted" / "Rejected"), we need 2 labels.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    print("\n✅ Model and Tokenizer downloaded successfully!")
    print("\n--- Tokenizer Info ---")
    print(tokenizer)
    
    print("\n--- Model Info ---")
    print(model.config)

    # Let's do a quick test to see if they work together
    print("\n--- Performing a quick test ---")
    test_text = "The plaintiff claims the defendant breached the contract."
    inputs = tokenizer(test_text, return_tensors="pt") # "pt" returns PyTorch tensors
    
    # The model will output 'logits', which are raw scores for each label
    with torch.no_grad(): # Disable gradient calculation for simple inference
        outputs = model(**inputs)
        logits = outputs.logits

    print(f"\nTest text: '{test_text}'")
    print(f"Model output (logits): {logits}")
    print("\nTest successful! The model can process text.")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")
    print("Please check your internet connection and the model name.")