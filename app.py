import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

labels = ["World", "Sports", "Business", "Sci/Tech"]

MODEL_PATH = "./model"
TOKENIZER_PATH = "./tokenizer"

# 🔥 FIX: fallback if model not found on Streamlit Cloud
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
else:
    st.warning("Local model not found → using base BERT (untrained)")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=4
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

st.title("📰 AG News Classifier (BERT)")

text = st.text_area("Enter news text")

if st.button("Predict"):
    if not text.strip():
        st.error("Type something bro 😭")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()

        st.success(f"Prediction: {labels[pred]}")
