import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------
# APP TITLE
# ----------------------------
st.title("📰 AG News Classifier (BERT)")

# ----------------------------
# LABELS
# ----------------------------
labels = ["World", "Sports", "Business", "Sci/Tech"]

# ----------------------------
# PRETRAINED FINE-TUNED MODEL
# ----------------------------
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
# UI
# ----------------------------
st.write("Enter a news headline or article below:")

text = st.text_area("News Input")

# Sample examples
samples = {
    "World": "The government announced new foreign policy reforms today.",
    "Sports": "The team won the championship in a thrilling final match.",
    "Business": "Stock markets surged after interest rate cuts.",
    "Sci/Tech": "Scientists developed a new AI model that beats humans in coding."
}

if st.button("Load Sample"):
    text = list(samples.values())[0]
    st.session_state["text"] = text

text = st.text_area("News Input", value=st.session_state.get("text", ""))

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        st.success(f"Prediction: {labels[pred]}")
        st.write("Confidence:", float(probs[0][pred]))
