def write_hyperparameters_to_file(params_set, file_path):
    with open(file_path, "w") as file:
        for param, value in params_set.items():
            file.write(f"{param}: {value}\n")
    print(f"Hyperparameters written to {file_path}")

