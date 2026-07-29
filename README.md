Financial Sentiment Analysis with FinBERT
A production-ready NLP pipeline for financial sentiment analysis using **FinBERT** and **PyTorch**.
This project implements an end-to-end workflow for fine-tuning a Transformer model (ProsusAI/finbert) on the Financial PhraseBank dataset, including data preprocessing, tokenization, model training, evaluation, inference, and comprehensive error analysis.


Project Overview
Financial news has a significant impact on stock prices and investment decisions. Automatically identifying the sentiment of financial texts helps analysts, investors, and trading systems process information more efficiently.
This project fine-tunes **FinBERT**, a Transformer model specialised for financial language, to classify financial news into three sentiment categories:
- 🔴 Negative
- ⚪ Neutral
- 🟢 Positive


Features
- Data preprocessing
- Train / Validation / Test split
- Hugging Face Tokenizer
- PyTorch DataLoader
- FinBERT fine-tuning
- Custom training loop (without Trainer API)
- Learning Rate Scheduler
- AdamW Optimizer
- Gradient Clipping
- Model Checkpointing
- Evaluation Metrics
- Error Analysis
- FasAPI


Project Structure

```text
Financial-sentiment-analysis/

├── data/
│   ├── raw/
│   ├── processed/
│   └── tokenized/
│
├── models/
│   └── best_model.pt
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_tokenizer_analysis.ipynb
│   └── 03_error_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── split.py
│   ├── tokenizer.py
│   ├── data_module.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── app.py
│   └── config.py
│
├── api/
│   ├── schemas/
│   ├── main/

├── requirements.txt
├── README.md
└── .gitignore
```


Dataset
**Financial PhraseBank**
The dataset contains financial news sentences manually annotated by financial experts.
Classes:
| Label | Meaning |
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |


Model

Model:
```
ProsusAI/finbert
```

Architecture:
```
Input Text
      │
      ▼
Tokenizer
      │
      ▼
Input Ids + Attention Mask
      │
      ▼
FinBERT Encoder
      │
      ▼
Classification Head
      │
      ▼
Logits
      │
      ▼
Softmax
      │
      ▼
Prediction
```


Training Configuration
| Parameter | Value |
|-----------|------
| Model | FinBERT |
| Epochs | 5 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |
| Scheduler | Linear Warmup |
| Max Length | 64 |


Evaluation Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix
- Classification Report


Error Analysis

The project includes a dedicated notebook for analysing model behaviour.
Analysis includes:
- Misclassified samples
- High-confidence mistakes
- Confusion Matrix
- Class-wise performance
- Prediction confidence distribution


Inference
Example:
Input:
```
The company reported record quarterly profits.
```
Output:
```
Positive
Confidence: 98.7%
```


Installation

Clone the repository:
```bash
Git clone https://github.com/maryamkhosravii/financial-sentiment-analysis.git
```
Install dependencies:

```bash
Pip install -r requirements.txt
```

Train the model:
```bash
Python src/train.py
```

Evaluate:
```bash
Python src/evaluate.py
```

Run inference:
```bash
Python src/inference.py
```

API Deployment:
Run:
uvicorn api.main:app --reload
Open Swagger documentation:
http://127.0.0.1:8000/docs


Technologies
- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- FastAPI


 Future Improvements:
- Hyperparameter Optimization (Optuna)
- Multi-model Benchmark (RoBERTa, DeBERTa, BERT)
- MLflow / Weights & Biases Integration
- Docker Support
- Hugging Face Spaces Deployment


Author:
Maryam Khosravi
GitHub:
https://github.com/maryamkhosravii

