# 🛠️ Install dependencies
!pip install transformers datasets scikit-learn -q

# 📦 Imports
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# 🚫 Disable Weights & Biases (W&B) logging if you don't want experiment tracking
os.environ["WANDB_DISABLED"] = "true"

# 🧮 Load & Prepare Dataset
fake = pd.read_csv("Fake.csv")     # Label 0 → Fake news
real = pd.read_csv("True.csv")     # Label 1 → Real news

# 🏷️ Add labels
fake['label'] = 0
real['label'] = 1

# 🔀 Combine and shuffle the dataset
# Reset index ensures clean DataFrame after shuffling
df = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)

# 🧾 Create a single text column for model input
df['content'] = df['title'] + " " + df['text']
df = df[['content', 'label']].dropna().reset_index(drop=True)

# ⏱️ Optional: Use a smaller sample for faster Colab runs
df = df.sample(10000, random_state=42).reset_index(drop=True)

# 🔤 Tokenization - Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(example):
    # Pad and truncate to BERT's max token length (512)
    return tokenizer(example['content'], truncation=True, padding='max_length', max_length=512)

# 📚 Convert DataFrame to HuggingFace dataset format
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)  # 80/20 train-test split

# 🔁 Apply tokenization to both train and test sets
tokenized = dataset.map(tokenize_function, batched=True)
tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 🤖 Load pre-trained BERT model for sequence classification (binary)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 📊 Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# ⚙️ Set training arguments
args = TrainingArguments(
    output_dir='./results',                  # Where to save the model
    per_device_train_batch_size=8,          # Tune based on RAM/GPU
    per_device_eval_batch_size=8,
    num_train_epochs=2,                     # Can try 3 or more
    weight_decay=0.01,                      # Helps regularization
    logging_steps=500,                      # Log every 500 steps
    save_steps=500,                         # Save checkpoint every 500 steps
    logging_dir=None                        # Disables W&B logging completely
)

# 🏋️ Create Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    compute_metrics=compute_metrics,
)

# 🚀 Start Training
trainer.train()

# 🧪 Evaluate the model
results = trainer.evaluate()

# 🧾 Print formatted results
print("\n\033[1m\u2728 Final Evaluation Results:\033[0m")
for key, value in results.items():
    if isinstance(value, float):
        print(f"{key:<20}: {value:.4f}")
