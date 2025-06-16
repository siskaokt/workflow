# Import library yang digunakan
import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Mengambil argumen dari command line
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='banknote_preprocessing.csv')
args = parser.parse_args()

# Load dataset
df = pd.read_csv("banknote_preprocessing.csv")

# Memisahkan fitur dan target
X = df.drop(columns="class")
y = df["class"]

# Train-Test split dengan rasio 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# GridSearch untuk tuning SVM
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

# Evaluasi dan logging manual
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

# Simpan model ke folder outputs
os.makedirs("outputs", exist_ok=True)
model_path = os.path.join("outputs", "model.pkl")
joblib.dump(best_model, model_path)

# Logging ke MLflow sebagai artifact
with mlflow.start_run(run_name="SVM - Tuning", nested=True):
    mlflow.log_param("model", "SVM")
    for param, val in grid.best_params_.items():
        mlflow.log_param(param, val)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    mlflow.log_artifact(model_path, artifact_path="model")