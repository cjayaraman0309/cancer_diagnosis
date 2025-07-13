with open("src/train.py", "w") as f:
    f.write('''import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv('data/preprocess/cancer_data_preprocessed.csv')
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

with mlflow.start_run() as run:
    # Set MLflow tags
    mlflow.set_tags({
        "created_by": "cjayaraman",
        "project": "cancer-diagnosis",
        "model_type": "RandomForestClassifier",
        "purpose": "ML pipeline demo"
    })

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(clf, "rf-model")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Run ID:", run.info.run_id)
''')
