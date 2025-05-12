# 📰 Fake News Classifier

This project aims to classify news articles as **Fake** or **Real** using two different approaches:

1. **Logistic Regression (baseline)**
2. **Fine-Tuned BERT (transformer-based model)**

---

## 🔧 Dataset

The dataset was compiled by combining:

* `Fake.csv` (label = 0)
* `True.csv` (label = 1)

Each entry combines the **title** and **text** into a single `content` column. A subset of 10,000 shuffled samples was used for experiments to keep runtime feasible on Google Colab.

The dataset used in this project is publicly available on Kaggle:
[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 📉 Baseline Model: Logistic Regression

A traditional machine learning model was trained on TF-IDF features extracted from the `content` column.

### 📈 Accuracy: `98.81%`

### 🧾 Classification Report:

```
               precision    recall  f1-score   support

           0       0.99      0.99      0.99      4710
           1       0.99      0.99      0.99      4270

    accuracy                           0.99      8980
   macro avg       0.99      0.99      0.99      8980
weighted avg       0.99      0.99      0.99      8980
```

### 🧩 Confusion Matrix:

```
               Predicted Fake | Predicted Real
Actual Fake        4646       |      64
Actual Real         43        |     4227
```

### ✅ Observations:

* Lightweight and fast
* Performs surprisingly well on clean text data
* Lacks contextual understanding (e.g., sarcasm, tone)

---

## 🤖 BERT Model: Fine-Tuned `bert-base-uncased`

Using HuggingFace Transformers, a BERT model was fine-tuned on the same dataset.

### ⚙️ Training Setup:

* Batch Size: 8
* Epochs: 2
* Max Length: 512 tokens
* Optimizer: AdamW (handled by `Trainer` API)

### 🧠 Final Evaluation Results:

```
eval_loss              : 0.0059
eval_accuracy          : 0.9990
eval_f1                : 0.9990
eval_precision         : 0.9990
eval_recall            : 0.9990
eval_runtime           : 55.48s
eval_samples_per_second: 36.05
eval_steps_per_second  : 4.51
```

### ✅ Observations:

* Excellent results — nearly perfect classification
* Much deeper understanding of context and semantics
* Requires significantly more time and memory to train

---

## 🔬 Comparison Table

| Metric               | Logistic Regression | BERT            |
| -------------------- | ------------------- | --------------- |
| Accuracy             | 98.81%              | 99.90%          |
| Precision            | 0.99                | 0.999           |
| Recall               | 0.99                | 0.999           |
| F1 Score             | 0.99                | 0.999           |
| Training Time        | Seconds             | \~25 minutes    |
| Interpretability     | High (white-box)    | Low (black-box) |
| Contextual Awareness | ❌                   | ✅               |

---

## 📌 Conclusion

Starting with logistic regression provided a strong baseline, showing that classical methods can be surprisingly effective on well-prepared datasets. However, BERT’s contextual power clearly provided a measurable improvement. If resources allow, transformer-based models are ideal for real-world deployment.

---

## 🛠️ Future Improvements

* Use dataset balancing or augmentation
* Try DistilBERT or RoBERTa for efficiency
* Experiment with longer training or hyperparameter tuning
* Add explainability (e.g., LIME or SHAP)

---

## 📚 Tools & Libraries

* HuggingFace Transformers
* scikit-learn
* PyTorch
* Pandas, Numpy
* Google Colab GPU
  
---
