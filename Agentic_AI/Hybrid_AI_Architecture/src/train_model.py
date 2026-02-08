import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from config import FEATURE_PATH, MODEL_PATH

df = pd.read_parquet(FEATURE_PATH)

y = df["label"]
X = df.drop(columns=["label"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

model = LGBMClassifier(
    objective="multiclass",
    num_class=3,
    n_estimators=500,
    learning_rate=0.04
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, pred))

print("\nClassification Report")
print(classification_report(y_test, pred))

joblib.dump(model, MODEL_PATH)
print("Model saved to:", MODEL_PATH)
