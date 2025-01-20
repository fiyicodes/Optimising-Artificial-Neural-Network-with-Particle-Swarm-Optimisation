import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import json
from neural_network import ANN
from pso import PSO
from utils import pack_weights, unpack_weights, sphere_function, rastrigin_function, rosenbrock_function, ackley_function
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)
STATIC_PATH = "./static"

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/optimize", methods=["POST"])
def optimize():
    # General ANN Configuration
    input_nodes = int(request.form.get("input_nodes"))
    output_nodes = int(request.form.get("output_nodes"))

    # Collect hidden layers dynamically
    hidden_layers = []
    for key in request.form.keys():
        if key.startswith("hidden_nodes_"):
            layer_index = key.split("_")[-1]
            nodes = int(request.form[key])
            activation = request.form.get(f"activation_{layer_index}", "relu")
            hidden_layers.append({"nodes": nodes, "activation": activation})

    # Construct ANN configuration
    ann_config = [{"nodes": input_nodes, "activation": None}] + hidden_layers + [{"nodes": output_nodes, "activation": "linear"}]

    # Print to debug the configuration
    print("ANN Configuration:", ann_config)

    # PSO Configuration
    pso_config = {
        "population_size": int(request.form.get("population_size")),
        "alpha": float(request.form.get("alpha")),
        "beta": float(request.form.get("beta")),
        "gamma": float(request.form.get("gamma")),
        "delta": float(request.form.get("delta")),
        "epsilon": float(request.form.get("epsilon")),
        "max_iter": int(request.form.get("max_iter")),
        "variant": request.form.get("variant"),
    }

    task = request.form.get("task")  # Ensure 'task' is retrieved from the form
    split_ratio = float(request.form.get("split_ratio", 0.2))
    random_seed = int(request.form.get("random_seed", 42))

    if task == "ann":
        # Run ANN optimization
        results = run_ann_optimization({
            "ann_config": ann_config,
            "pso_config": pso_config,
            "split_ratio": split_ratio,
            "random_seed": random_seed,
        })

    elif task == "benchmark":
        # Handle benchmark function optimization
        benchmark_function = request.form.get("benchmark_function")
        benchmark_map = {
            "sphere": sphere_function,
            "rastrigin": rastrigin_function,
            "rosenbrock": rosenbrock_function,
            "ackley": ackley_function,
        }
        if benchmark_function not in benchmark_map:
            raise ValueError(f"Unsupported benchmark function: {benchmark_function}")

        benchmark_func = benchmark_map[benchmark_function]
        results = run_benchmark_optimization({
            "pso_config": pso_config,
            "benchmark_function": benchmark_func,
        })
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Print results for debugging
    print("Results:", results)

    # Render the results page with the computed results
    return render_template("results.html", results=results)

# Render the optimization form for GET requests
@app.route("/optimize", methods=["GET"])
def render_optimize_form():
    return render_template("optimize.html")

def run_benchmark_optimization(config):
    benchmark_function = config["benchmark_function"]
    pso_config = config["pso_config"]
    population_size = pso_config["population_size"]
    alpha = pso_config["alpha"]
    beta = pso_config["beta"]
    gamma = pso_config["gamma"]
    delta = pso_config["delta"]
    epsilon = pso_config["epsilon"]
    max_iter = pso_config["max_iter"]
    variant = pso_config["variant"]
    dim = 10  # Dimensionality of the optimization problem

    pso = PSO(benchmark_function, population_size, dim, alpha, beta, gamma, delta, epsilon, max_iter, variant=variant)
    best_position, best_loss = pso.optimize()

    # Save loss history plot
    plot_path = os.path.join(STATIC_PATH, "loss_history.png")
    plt.figure()
    plt.plot(pso.loss_history)
    plt.title(f"PSO Optimization of {benchmark_function.__name__}")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(plot_path)  # Save plot as a static image
    plt.close()  # Ensure the figure is closed to avoid GUI conflicts

    return {
        "type": "benchmark",
        "function": benchmark_function.__name__,
        "best_position": best_position.tolist(),
        "best_loss": best_loss,
        "plots": {
            "loss_history": plot_path,
        },
    }


def run_ann_optimization(config):
    dataset_path = "concrete_data.csv"  # Predefined dataset path
    ann_config = config["ann_config"]
    task = "regression"  # Explicitly set task as "regression"
    pso_config = config["pso_config"]
    split_ratio = config["split_ratio"]
    random_seed = config["random_seed"]

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

    # Define the fitness function for PSO
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

    # Training loss after optimization
    y_pred_train = ann.forward(X_train_scaled)
    train_loss_scaled = ann.loss(y_train_scaled, y_pred_train)
    y_pred_train_original = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
    train_loss_original = ann.loss(y_train.values, y_pred_train_original)

    # Test loss after optimization
    y_pred_test = ann.forward(X_test_scaled)
    test_loss_scaled = ann.loss(y_test_scaled, y_pred_test)
    y_pred_test_original = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
    test_loss_original = ann.loss(y_test.values, y_pred_test_original)

    best_compressive_strength = np.max(y_pred_test_original)

    # Save the true vs predicted plot for training data
    true_vs_pred_train_path = os.path.join(STATIC_PATH, "true_vs_pred_train.png")
    plt.figure()
    plt.scatter(y_train, y_pred_train_original, alpha=0.7, color="orange")
    plt.title("True vs Predicted Compressive Strength (Training Data)")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.grid()
    plt.savefig(true_vs_pred_train_path)
    plt.close()

    # Save the true vs predicted plot for test data
    true_vs_pred_test_path = os.path.join(STATIC_PATH, "true_vs_pred_test.png")
    plt.figure()
    plt.scatter(y_test, y_pred_test_original, alpha=0.7)
    plt.title("True vs Predicted Compressive Strength (Test Data)")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.grid()
    plt.savefig(true_vs_pred_test_path)
    plt.close()

    # Save the loss history plot
    loss_history_path = os.path.join(STATIC_PATH, "loss_history_ann.png")
    plt.figure()
    plt.plot(pso.loss_history)
    plt.title("PSO Loss History for ANN Optimization")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(loss_history_path)
    plt.close()

    # Save the error distribution plot for test data
    errors = y_test.values - y_pred_test_original
    error_dist_path = os.path.join(STATIC_PATH, "error_distribution_test.png")
    plt.figure()
    plt.hist(errors, bins=20, alpha=0.7, color="red")
    plt.title("Error Distribution (Test Data)")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig(error_dist_path)
    plt.close()

    # Return results
    return {
        "type": "ann",
        "best_loss": best_loss,
        "train_loss_scaled": train_loss_scaled,
        "train_loss_original": train_loss_original,
        "test_loss_scaled": test_loss_scaled,
        "test_loss_original": test_loss_original,
        "best_compressive_strength": best_compressive_strength,
        "plots": {
            "true_vs_pred_train": true_vs_pred_train_path,
            "true_vs_pred_test": true_vs_pred_test_path,
            "loss_history": loss_history_path,
            "error_distribution_test": error_dist_path,
        },
    }

if __name__ == "__main__":
    if not os.path.exists(STATIC_PATH):
        os.makedirs(STATIC_PATH)
    app.run(debug=True, port=5001)
