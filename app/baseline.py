
import pandas as pd
import time
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================
# 1. CARGAR LOS DATOS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

train_df = pd.read_csv(
    BASE_DIR / "data" / "train.tsv",
    sep="\t"
)

test_df = pd.read_csv(
    BASE_DIR / "data" / "test.tsv",
    sep="\t"
)

print("Datos cargados correctamente")
print("Entrenamiento:", train_df.shape)
print("Prueba:", test_df.shape)

# =====================================
# 2. PREPARAR LOS DATOS
# =====================================

train_df = train_df[["text", "label"]].dropna().copy()
test_df = test_df[["text", "label"]].dropna().copy()

mapa = {
    "human": 0,
    "generated": 1
}

train_df["label"] = train_df["label"].map(mapa)
test_df["label"] = test_df["label"].map(mapa)

train_df = train_df.dropna(subset=["label"])
test_df = test_df.dropna(subset=["label"])

train_df["label"] = train_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)

# =====================================
# 3. DIVIDIR ENTRENAMIENTO Y VALIDACIÓN
# =====================================

train_part, val_part = train_test_split(
    train_df,
    test_size=0.10,
    random_state=42,
    stratify=train_df["label"]
)

X_train = train_part["text"].astype(str)
y_train = train_part["label"]

X_val = val_part["text"].astype(str)
y_val = val_part["label"]

X_test = test_df["text"].astype(str)
y_test = test_df["label"]

print("\nEntrenamiento:", len(X_train))
print("Validación:", len(X_val))
print("Prueba:", len(X_test))

# =====================================
# 4. CREAR EL MODELO BASE
# =====================================

modelo = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100000,
        min_df=2,
        sublinear_tf=True
    )),

    ("clasificador", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# =====================================
# 5. ENTRENAR EL MODELO
# =====================================

inicio = time.time()

modelo.fit(X_train, y_train)

tiempo_entrenamiento = time.time() - inicio

print("\nModelo entrenado correctamente")

# =====================================
# 6. EVALUAR VALIDACIÓN
# =====================================

pred_val = modelo.predict(X_val)

f1_val = f1_score(
    y_val,
    pred_val,
    average="macro"
)

print("\n===== VALIDACIÓN =====")
print("F1 macro:", f1_val)

# =====================================
# 7. EVALUAR PRUEBA
# =====================================

inicio = time.time()

pred_test = modelo.predict(X_test)

tiempo_prueba = time.time() - inicio

print("\n===== RESULTADOS DE PRUEBA =====")

print("F1 macro:",
      f1_score(y_test, pred_test, average="macro"))

print("Accuracy:",
      accuracy_score(y_test, pred_test))

print("\nReporte de clasificación:")

print(classification_report(
    y_test,
    pred_test,
    labels=[0, 1],
    target_names=["Humano", "Máquina"]
))

print("Matriz de confusión:")
print(confusion_matrix(y_test, pred_test))

# =====================================
# 8. MOSTRAR TIEMPOS
# =====================================

print("\n===== TIEMPOS =====")

print("Entrenamiento:",
      round(tiempo_entrenamiento, 2), "segundos")

print("Prueba:",
      round(tiempo_prueba, 2), "segundos")

# =====================================
# 9. GUARDAR EL MODELO
# =====================================

carpeta_modelos = BASE_DIR / "models"
carpeta_modelos.mkdir(exist_ok=True)

joblib.dump(
    modelo,
    carpeta_modelos / "baseline_tfidf_logreg.pkl"
)

print("\nModelo guardado correctamente")
