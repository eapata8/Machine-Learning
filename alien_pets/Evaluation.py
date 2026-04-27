import os
import tarfile
import json
import pathlib
import joblib
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score


if __name__ == "__main__":

    model_tar_path = "/opt/ml/processing/model/model.tar.gz"

    with tarfile.open(model_tar_path) as tar:
        tar.extractall(path="/opt/ml/processing/model")

    model = joblib.load("/opt/ml/processing/model/model.joblib")

    test_data = pd.read_csv("/opt/ml/processing/test/test.csv")

    X_test = test_data.drop(columns=["petal_length"])
    y_test = test_data["petal_length"]

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2: {r2:.4f}")

    report_dict = {
        "regression_metrics": {
            "mse": {"value": mse},
            "r2": {"value": r2}
        }
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/evaluation.json", "w") as f:
        json.dump(report_dict, f)
      
