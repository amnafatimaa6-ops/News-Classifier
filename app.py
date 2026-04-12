import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------
# PAGE CONFIG (PRO LOOK)
# ----------------------------
st.set_page_config(
    page_title="AG News Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AG News Classifier (BERT)")
st.caption("Scholarship Project | NLP | Transformer-based Text Classification")

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
# SAMPLES
# ----------------------------
samples = {
    "World 🌍": "The government announced new foreign policy reforms today.",
    "Sports ⚽": "The team won the championship after a dramatic final match.",
    "Business 💼": "Stock markets surged after interest rate cuts by the central bank.",
    "Sci/Tech 🤖": "Scientists developed a new AI model that outperforms humans in coding."
}

# ----------------------------
# INPUT MODE
# ----------------------------
st.markdown("### 📌 Input Section")

mode = st.radio("Choose input mode:", ["✍️ Write my own", "📚 Use sample"])

text = ""

if mode == "📚 Use sample":
    choice = st.selectbox("Select a sample", list(samples.keys()))
    text = samples[choice]
    st.info(text)
else:
    text = st.text_area("Enter your text here", height=150)

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("🚀 Predict Category"):
    if not text.strip():
        st.warning("Please enter text first.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        pred = torch.argmax(probs).item()

        # ----------------------------
        # RESULT SECTION
        # ----------------------------
        st.markdown("## 🎯 Prediction Result")

        st.success(f"Category: **{labels[pred]}**")

        confidence = float(probs[pred])

        st.markdown("### 📊 Confidence Score")
        st.progress(confidence)
        st.write(f"Confidence: **{confidence:.3f}**")

        # ----------------------------
        # TOP 2 PREDICTIONS (PORTFOLIO LEVEL)
        # ----------------------------
        top2 = torch.topk(probs, 2)

        st.markdown("### 🧠 Model Reasoning (Top Predictions)")

        for i in range(2):
            idx = top2.indices[i].item()
            score = float(top2.values[i])
            st.write(f"🔹 {labels[idx]} → {score:.3f}")
