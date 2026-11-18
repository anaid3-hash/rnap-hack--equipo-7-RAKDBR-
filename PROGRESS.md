## CP2 - 05-11-2026
- What we have done: Loaded and split the dataset into train, validation, and test sets. Trained a baseline model using TF-IDF and Logistic Regression, and evaluated a pre-trained GPT-2 model for perplexity without fine-tuning. Measured F1 score, Accuracy, and processing times for both models. Documented sections 1, 2, and 3 in the technical report notebook.
- What we haven't done yet: Transformer fine-tuning and v1 interface development (Phase 2).
- Next step: Build the fine-tuned Transformer model and start designing the user interface.
- Who did what: All team members collaborated on code implementation, evaluation, and documentation.

## Checkpoint 3 (CP3) - Streamlit Interface & Model Integration
- **Status:** Completed
- **Summary:** Developed the interactive web interface using Streamlit (`app.py`) and integrated the trained baseline classification model (TF-IDF + Logistic Regression) to allow real-time text evaluation (Human vs. Machine).
- **Key Deliverables:** Functional user interface, model serialization/loading pipeline, and dependency management (`requirements.txt`).

## CP4 - 17-11-2026
- Successfully configured and tested the Streamlit user interface (app/app.py).
- Resolved environment dependencies and installed required packages (joblib, scikit-learn, streamlit) inside the virtual environment (venv).
- Successfully integrated the trained machine learning model (baseline_tfidf_logreg.pkl) into the web application for real-time predictions.
