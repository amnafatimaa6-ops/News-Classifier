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
# SAMPLE DATA
# ----------------------------
samples = {
    "World 🌍": "The government announced new foreign policy reforms today.",
    "Sports ⚽": "The team won the championship after a dramatic final match.",
    "Business 💼": "Stock markets surged after interest rate cuts by the central bank.",
    "Sci/Tech 🤖": "Scientists developed a new AI model that outperforms humans in coding."
}

# ----------------------------
# USER CHOICE
# ----------------------------
option = st.selectbox(
    "Choose a sample or write your own",
    ["Write my own"] + list(samples.keys())
)

# ----------------------------
# TEXT INPUT LOGIC
# ----------------------------
if option == "Write my own":
    text = st.text_area("Enter your news text")
else:
    text = samples[option]
    st.text_area("Selected sample", value=text, height=120)

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
