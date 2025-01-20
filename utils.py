import numpy as np

def pack_weights(weights, biases):
    return np.concatenate([w.flatten() for w in weights] + [b.flatten() for b in biases])

def unpack_weights(flat_params, layers_config):
    weights = []
    biases = []
    idx = 0
    for i in range(1, len(layers_config)):
        input_dim = layers_config[i - 1]['nodes']
        output_dim = layers_config[i]['nodes']

        w_size = input_dim * output_dim
        weights.append(flat_params[idx:idx + w_size].reshape(input_dim, output_dim))
        idx += w_size

        biases.append(flat_params[idx:idx + output_dim])
        idx += output_dim

    return weights, biases

# Benchmark functions for PSO
def sphere_function(x):
    return np.sum(x ** 2)

def rastrigin_function(x):
    A = 10
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

def rosenbrock_function(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def ackley_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)
