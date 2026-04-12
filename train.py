from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import torch
import os

dataset = load_dataset("ag_news")

labels = ["World", "Sports", "Business", "Sci/Tech"]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

train_ds = dataset["train"].map(tokenize, batched=True)
test_ds = dataset["test"].map(tokenize, batched=True)

cols = ["input_ids", "attention_mask", "label"]
train_ds.set_format("torch", columns=cols)
test_ds.set_format("torch", columns=cols)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# TRAIN
for epoch in range(2):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# SAVE MODEL
os.makedirs("saved_model", exist_ok=True)
model.save_pretrained("saved_model/model")
tokenizer.save_pretrained("saved_model/tokenizer")

print("Model saved successfully 🚀")
