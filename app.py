import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AG News Research Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AG News Classification Research Dashboard")
st.caption("Transformer-based NLP System | Evaluation + Explainability + Inference")

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
# SIDEBAR NAVIGATION
# ----------------------------
page = st.sidebar.radio(
    "Navigation",
    ["🔮 Predict", "📊 Evaluation Demo"]
)

# =========================================================
# 🔮 PREDICTION PAGE
# =========================================================
if page == "🔮 Predict":

    st.header("News Classification")

    samples = {
        "World": "The government announced new foreign policy reforms today.",
        "Sports": "The team won the championship after a dramatic final match.",
        "Business": "Stock markets surged after interest rate cuts.",
        "Sci/Tech": "Scientists developed a new AI model that beats humans in coding."
    }

    mode = st.radio("Input Mode:", ["Write text", "Use sample"])

    if mode == "Use sample":
        choice = st.selectbox("Choose sample", list(samples.keys()))
        text = samples[choice]
        st.text_area("Input", value=text, height=120, disabled=True)
    else:
        text = st.text_area("Enter text", height=150)

    if st.button("Predict"):

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        pred = torch.argmax(probs).item()

        st.success(f"Prediction: {labels[pred]}")
        st.metric("Confidence", f"{float(probs[pred]):.3f}")

        st.subheader("Class Probabilities")

        prob_df = pd.DataFrame({
            "Class": labels,
            "Score": [float(p) for p in probs]
        })

        st.bar_chart(prob_df.set_index("Class"))

        st.subheader("Top Predictions")
        topk = torch.topk(probs, 2)

        for i in range(2):
            idx = topk.indices[i].item()
            st.write(f"🔹 {labels[idx]} → {float(topk.values[i]):.3f}")

# =========================================================
# 📊 EVALUATION PAGE
# =========================================================
elif page == "📊 Evaluation Demo":

    st.header("Model Evaluation (Simulated Demo)")

    st.write("This section demonstrates how research evaluation is presented.")

    true = np.random.randint(0, 4, 200)
    pred = np.random.randint(0, 4, 200)

    acc = (true == pred).mean()

    st.metric("Accuracy (demo)", f"{acc:.3f}")

    st.subheader("Classification Report")
    st.text(classification_report(true, pred, target_names=labels))

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(true, pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
    st.pyplot(fig)
