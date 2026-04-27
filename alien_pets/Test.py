outputs = sklearn_processor.latest_job.describe()["ProcessingOutputConfig"]["Outputs"]

output_paths = {
    output["OutputName"]: output["S3Output"]["S3Uri"]
    for output in outputs
}

preprocessed_training_data = output_paths["train"]
preprocessed_test_data = output_paths["test"]

print("Training data S3 URI:", preprocessed_training_data)
print("Test data S3 URI:", preprocessed_test_data)
