import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from neural_network import ANN
from pso import PSO
from utils import pack_weights, unpack_weights, sphere_function, rastrigin_function, rosenbrock_function, ackley_function


def main():
    # Load configuration
    try:
        with open("config.json", "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        print("Error: Configuration file 'config.json' not found.")
        return

    # Extract configuration parameters
    dataset_path = config.get("dataset", None)
    ann_config = config.get("ann_config", {}).get("layers", [])
    task = config.get("ann_config", {}).get("task", "regression")
    pso_config = config.get("pso_config", {})
    benchmark_function_name = config.get("benchmark_function", None)
    split_ratio = config.get("split_ratio", 0.2)
    random_seed = config.get("random_seed", 42)

    # Benchmark function mapping
    benchmark_map = {
        "sphere": sphere_function,
        "rastrigin": rastrigin_function,
        "rosenbrock": rosenbrock_function,
        "ackley": ackley_function,
    }

    if benchmark_function_name in benchmark_map:
        # Run benchmark function optimization
        benchmark_function = benchmark_map[benchmark_function_name]
        run_benchmark_optimization(benchmark_function, pso_config)
    elif dataset_path:
        # Run ANN optimization
        run_ann_optimization(dataset_path, ann_config, task, pso_config, split_ratio, random_seed)
    else:
        print("Error: Either 'dataset' or 'benchmark_function' must be specified in config.json.")
        return


def run_benchmark_optimization(benchmark_function, pso_config):
    print(f"Optimizing benchmark function: {benchmark_function.__name__}")

    # Initialize PSO
    dim = 10  # Dimensionality of the optimization problem
    pso = PSO(
        benchmark_function,
        pso_config["population_size"],
        dim,
        pso_config["alpha"],
        pso_config["beta"],
        pso_config["gamma"],
        pso_config["delta"],
        pso_config["epsilon"],
        pso_config["max_iter"],
        variant=pso_config["variant"],
    )
    best_position, best_loss = pso.optimize()

    print(f"Best position: {best_position}")
    print(f"Best loss: {best_loss}")

    # Plot loss history
    plt.figure()
    plt.plot(pso.loss_history)
    plt.title(f"PSO Optimization of {benchmark_function.__name__}")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()


def run_ann_optimization(dataset_path, ann_config, task, pso_config, split_ratio, random_seed):
    print(f"Optimizing ANN for task: {task}")

    # Load dataset
    data = pd.read_csv(dataset_path)
    X = data.drop(columns=["concrete_compressive_strength"])
    y = data["concrete_compressive_strength"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_seed)

    # Normalize the dataset
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    # Initialize ANN
    ann = ANN(ann_config, task=task)

    def fitness_function_scaled(flat_params):
        weights, biases = unpack_weights(flat_params, ann_config)
        ann.set_weights(weights, biases)
        y_pred = ann.forward(X_train_scaled)
        return ann.loss(y_train_scaled, y_pred)

    # PSO Parameters
    dim = sum(w.size for w in ann.weights) + sum(b.size for b in ann.biases)
    pso = PSO(
        fitness_function_scaled,
        pso_config["population_size"],
        dim,
        pso_config["alpha"],
        pso_config["beta"],
        pso_config["gamma"],
        pso_config["delta"],
        pso_config["epsilon"],
        pso_config["max_iter"],
        variant=pso_config["variant"],
    )
    best_params, best_loss = pso.optimize()

    print(f"Best loss on training data (scaled): {best_loss}")

    # Set the ANN to the best parameters and evaluate
    best_weights, best_biases = unpack_weights(best_params, ann_config)
    ann.set_weights(best_weights, best_biases)

    # Evaluate on test data
    y_pred_test = ann.forward(X_test_scaled)
    test_loss_scaled = ann.loss(y_test_scaled, y_pred_test)
    print(f"MSE on test data (scaled): {test_loss_scaled}")

    y_pred_test_original = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
    test_loss_original = ann.loss(y_test.values, y_pred_test_original)
    print(f"MSE on test data (original scale): {test_loss_original}")

    best_compressive_strength = np.max(y_pred_test_original)
    print(f"Best predicted compressive strength: {best_compressive_strength}")

    # Plot results
    plt.figure()
    plt.plot(pso.loss_history)
    plt.title("PSO Training Loss Over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

    plt.figure()
    plt.scatter(y_test, y_pred_test_original, alpha=0.7)
    plt.title("True vs Predicted Compressive Strength")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.grid()
    plt.show()

    errors = y_test.values - y_pred_test_original
    plt.figure()
    plt.hist(errors, bins=20, alpha=0.7, color="red")
    plt.title("Error Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
