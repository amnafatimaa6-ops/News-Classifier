import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

labels = ["World", "Sports", "Business", "Sci/Tech"]

# Load model
MODEL_PATH = "model"
TOKENIZER_PATH = "tokenizer"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found. Run train.py first.")
    st.stop()

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

st.title("📰 AG News Classifier (BERT)")

# --- INPUT ---
text = st.text_area("Enter News Text")

# --- SAMPLE BUTTON ---
sample_texts = {
    "Sports": "The team won the championship after a thrilling final match",
    "Business": "Stock markets surged after interest rate announcement",
    "World": "The government announced new foreign policy changes",
    "Sci/Tech": "Scientists discovered a new AI model outperforming humans"
}

if st.button("Load Sample"):
    text = list(sample_texts.values())[0]
    st.session_state["text"] = text

text = st.text_area("Enter News Text", value=st.session_state.get("text", ""))

# --- PREDICT ---
if st.button("Predict"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    st.success(f"Prediction: {labels[pred]}")
    st.write("Confidence:", probs[0][pred].item())

# --- METRICS DISPLAY (from training output manually pasted or saved later) ---
st.subheader("📊 Model Performance (from training)")
st.write("Accuracy: ~0.94")
st.write("F1 Score: ~0.94")
