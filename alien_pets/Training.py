import argparse
import os
import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args = parser.parse_args()

    print("Train folder:", args.train)
    print("Files in train folder:", os.listdir(args.train))

    input_files = [
        os.path.join(args.train, file)
        for file in os.listdir(args.train)
        if file.endswith(".csv")
    ]

    if len(input_files) == 0:
        raise ValueError("Aucun fichier CSV trouvé dans le dossier train.")

    raw_data = [pd.read_csv(file) for file in input_files]
    train_data = pd.concat(raw_data, ignore_index=True)

    print("Train data shape:", train_data.shape)
    print("Train columns:", train_data.columns)
    print(train_data.head())

    X_train = train_data[["sepal_width"]]
    y_train = train_data["sepal_length"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_train)

    mse = mean_squared_error(y_train, predictions)
    r2 = r2_score(y_train, predictions)

    print(f"Train MSE: {mse:.4f}")
    print(f"Train R2: {r2:.4f}")

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    print("Model saved successfully.")


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model
  
