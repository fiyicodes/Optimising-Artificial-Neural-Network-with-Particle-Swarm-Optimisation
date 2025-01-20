import numpy as np

class ANN:
    def __init__(self, layers_config, task="regression"):
        """
        Initialize the ANN.
        layers_config: List of layers with nodes and activation functions.
        task: "regression", "binary_classification", or "multi_class_classification".
        """
        self.layers = []
        self.activations = []
        self.weights = []
        self.biases = []
        self.task = task  # Determine output activation and loss function
        
        for i in range(1, len(layers_config)):
            input_dim = layers_config[i - 1]['nodes']
            output_dim = layers_config[i]['nodes']
            activation = layers_config[i]['activation']

            # Initialize weights and biases randomly
            self.weights.append(np.random.randn(input_dim, output_dim))
            self.biases.append(np.random.randn(output_dim))
            self.activations.append(activation)

    def _activate(self, x, activation):
        """
        Apply the specified activation function.
        """
        if activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        elif activation == 'linear':
            return x
        elif activation is None:
            return x
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, X):
        """
        Perform a forward pass through the network.
        """
        output = X
        for W, b, activation in zip(self.weights, self.biases, self.activations):
            output = np.dot(output, W) + b  # Linear transformation
            output = self._activate(output, activation)  # Apply activation
        return output

    def loss(self, y_true, y_pred):
        """
        Compute the loss based on the task type.
        """
        if self.task == "regression":
            # Mean Squared Error for regression
            return np.mean((y_true - y_pred) ** 2)
        elif self.task == "binary_classification":
            # Binary cross-entropy loss for binary classification
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.task == "multi_class_classification":
            # Categorical cross-entropy for multi-class classification
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def set_weights(self, weights, biases):
        """
        Set the weights and biases of the network.
        """
        self.weights = weights
        self.biases = biases
