import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Cargar el conjunto de prueba
test_df = pd.read_csv("app/data/test.tsv", sep="\t") 

X_test = test_df['text']
y_true_raw = test_df['label']

# 2. Convertir las etiquetas de texto ('human', 'generated') a números (0 y 1)
# (Ajusta 'generated' o 'human' según cuáles sean los nombres exactos en tu archivo)
y_test = y_true_raw.apply(lambda x: 1 if str(x).lower() in ['generated', 'ai', '1'] else 0)

# 3. Cargar tu modelo guardado
model = joblib.load("app/models/baseline_tfidf_logreg.pkl")

# 4. Hacer predicciones
y_pred = model.predict(X_test)

# 5. Calcular métricas clave sin errores de tipo
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))