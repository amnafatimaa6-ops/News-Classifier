import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# ----------------------------
# PAGE CONFIG (PRO LEVEL)
# ----------------------------
st.set_page_config(
    page_title="AG News Classifier | NLP Project",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AG News Classifier (BERT)")
st.caption("Scholarship Portfolio Project | NLP | Transformer-based Text Classification")

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
# SIDEBAR INFO (PORTFOLIO LOOK)
# ----------------------------
st.sidebar.title("📊 Project Info")
st.sidebar.write("NLP Text Classification using BERT")
st.sidebar.write("Dataset: AG News")
st.sidebar.write("Model: Fine-tuned Transformer")
st.sidebar.write("Classes: 4")

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
# INPUT SECTION
# ----------------------------
st.markdown("## 📌 Input Section")

mode = st.radio("Choose input mode:", ["✍️ Write your own", "📚 Use sample"])

text = ""

if mode == "📚 Use sample":
    choice = st.selectbox("Select a sample", list(samples.keys()))
    text = samples[choice]
    st.text_area("Selected text", value=text, height=120, disabled=True)
else:
    text = st.text_area("Enter your text", height=150)

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
        st.metric(label="Confidence Score", value=f"{confidence:.3f}")

        # ----------------------------
        # VISUAL PROBABILITY TABLE
        # ----------------------------
        st.markdown("## 📊 Class Probabilities")

        data = {
            "Class": labels,
            "Probability": [float(p) for p in probs]
        }

        df = pd.DataFrame(data)
        st.bar_chart(df.set_index("Class"))

        # ----------------------------
        # TOP-K ANALYSIS
        # ----------------------------
        st.markdown("## 🧠 Model Interpretation (Top Predictions)")

        top2 = torch.topk(probs, 2)

        for i in range(2):
            idx = top2.indices[i].item()
            score = float(top2.values[i])
            st.write(f"🔹 **{labels[idx]}** → {score:.3f}")

        # ----------------------------
        # SIMPLE INSIGHT BOX
        # ----------------------------
        st.info(
            "The model uses contextual embeddings from BERT to classify news text "
            "based on semantic patterns learned from the AG News dataset."
        )

# ----------------------------
# FOOTER (PORTFOLIO SIGNATURE)
# ----------------------------
st.markdown("---")
st.markdown("📌 Built using PyTorch + Transformers + Streamlit")
st.markdown("🎓 Suitable for ML Portfolio / Scholarship Submission")
