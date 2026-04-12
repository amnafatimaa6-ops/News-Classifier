import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
MODEL_PATH = "saved_model/model"
TOKENIZER_PATH = "saved_model/tokenizer"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

labels = ["World", "Sports", "Business", "Sci/Tech"]

st.title("📰 AG News Classifier (BERT)")
st.write("Paste a news headline or article and I’ll classify it.")

text = st.text_area("Enter text here:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Enter some text first bro 😭")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()

        st.success(f"Prediction: {labels[pred]}")
