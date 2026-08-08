import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_PATH = "diabetes.csv"
MODEL_PATH = "diabetes_model.pkl"

IMPUTE_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in IMPUTE_COLUMNS:
        median_value = df.loc[df[col] > 0, col].median()
        df.loc[df[col] == 0, col] = median_value
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    return pipeline


def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def main() -> None:
    df = load_data(DATA_PATH)
    df = clean_data(df)
    model = train_and_evaluate(df)
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()
