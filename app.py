import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("📰 AG News Classifier (BERT)")

labels = ["World", "Sports", "Business", "Sci/Tech"]

MODEL_NAME = "textattack/bert-base-uncased-ag-news"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ----------------------------
# SAMPLES (ONLY FOR QUICK TESTING)
# ----------------------------
samples = {
    "Sample 1": "The government announced new foreign policy reforms today.",
    "Sample 2": "The team won the championship after a dramatic final match.",
    "Sample 3": "Stock markets surged after interest rate cuts.",
    "Sample 4": "Scientists developed a new AI model that beats humans in coding."
}

# ----------------------------
# INPUT MODE
# ----------------------------
mode = st.radio("Choose input mode:", ["Write my own", "Use sample"])

text = ""

if mode == "Use sample":
    choice = st.selectbox("Pick a sample", list(samples.keys()))
    text = samples[choice]
    st.info(text)
else:
    text = st.text_area("Enter your news text")

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Predict"):
    if not text.strip():
        st.warning("Write something first.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        st.success(f"Prediction: {labels[pred]}")
        st.write("Confidence:", round(float(probs[0][pred]), 4))
