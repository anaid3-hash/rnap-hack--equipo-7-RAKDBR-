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

## Phase 2 (CP4) Results & Evaluation
* **Model Used:** TF-IDF + Logistic Regression (Baseline integrated into Streamlit app)
* **Test Accuracy:** 0.5991
* **F1-Score:** 0.5039
* **Error Analysis:** Documented 5 specific failure cases (Language barriers, colloquialisms, text length, formal/academic styles, and spelling errors) inside the official technical notebook.

## Live Demo
You can test and interact with the deployed application directly here:
[Open Streamlit App](https://f6yqhyx2sw8twfsvrlbkes.streamlit.app/)

## 6. Conclusions and Results Analysis

In this project, a Natural Language Processing (NLP) classifier was developed and implemented using a `TfidfVectorizer` pipeline and a `LogisticRegression` model to distinguish between human-authored texts and AI-generated texts (`'human'` vs. `'generated'`).

Based on the evaluation performed on the test dataset (`test.tsv`), the following key conclusions were reached:

* **Overall Performance:** The confusion matrix demonstrates a high accuracy rate in the human-authored class, successfully classifying over 10,700 samples. This indicates that the model robustly recognizes the linguistic patterns characteristic of human writing.
* **Area Under the ROC Curve (AUC):** The generated ROC curve reflects the model's effectiveness in discriminating between both classes across different probability thresholds. The visual behavior confirms that the model significantly outperforms the random baseline, showing competitive and stable performance.
* **Challenges and Limitations:** A margin of false positives and false negatives was observed in AI-generated texts. This occurs because modern language models closely mimic the structure, perplexity, and burstiness of human writing, posing a natural challenge for linear TF-IDF classifiers.
* **Practical Implications:** Integrating this model into the web interface deployed on Streamlit enables fast and accessible detection of synthetic content, fulfilling all technical and practical requirements of the project.

## Team
- **Team 7 - RAKDBR**
