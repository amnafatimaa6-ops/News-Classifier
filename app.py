import streamlit as st
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

st.title("📰 AG News Classifier (Auto-Trained BERT)")

labels = ["World", "Sports", "Business", "Sci/Tech"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def train_model():
    dataset = load_dataset("ag_news")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    train_ds = dataset["train"].map(tokenize, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=4
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1):  # keep it small for Streamlit
        model.train()
        for i, batch in enumerate(train_loader):
            if i > 50:  # LIMIT training (important for cloud)
                break

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels_batch)

            loss.backward()
            optimizer.step()

    return model, tokenizer

model, tokenizer = train_model()
model.eval()

text = st.text_area("Enter news text")

if st.button("Predict"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    st.success(f"Prediction: {labels[pred]}")
