# Human or Machine? - AI-Generated Text Detector

## Description
This project is developed for the Deep Learning course. Since the dataset used for training and evaluation is in English, the system is designed to distinguish between human-written and machine-generated English text using Natural Language Processing (NLP) techniques.

We implemented and compared two approaches:
1. **Baseline Model:** TF-IDF vectorization combined with Logistic Regression.
2. **Pre-trained Transformer:** GPT-2 perplexity-based classification (zero-shot / without fine-tuning).

## Model Comparison (Phase 1 Results)

| Model | F1 Score | Accuracy | Approach |
| :--- | :---: | :---: | :--- |
| **TF-IDF + Logistic Regression** | 0.5215 | 0.5869 | Statistical / Classical ML Baseline |
| **GPT-2 (Perplexity)** | **0.6730** | **0.6790** | Pre-trained Language Model (Zero-shot) |

## Repository Structure
- `notebooks/`: Contains the main technical report and experimental notebooks (`entrenamiento.ipynb`).
- `app/`: Source code for the application and model scripts (`app.py`, `baseline.py`, `gpt2_detector.py`).
- `data/`: Sample datasets and data references (`train.tsv`, `test.tsv`).
- `requirements.txt`: Python dependencies required to run the project.

## Team
- **Team 7 - RAKDBR**
