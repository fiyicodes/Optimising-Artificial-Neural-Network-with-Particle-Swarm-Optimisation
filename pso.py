import numpy as np

class PSO:
    def __init__(self, fitness_function, population_size, dim, alpha, beta, gamma, delta, epsilon, max_iter, variant="standard"):
        """
        Initialize PSO.
        variant: "standard", "inertia", or "constricted".
        """
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.variant = variant

        self.positions = np.random.uniform(-1, 1, (population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(population_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.loss_history = []

        if variant == "inertia":
            self.inertia_weight = 0.9  # Start with high inertia
        elif variant == "constricted":
            self.chi = 0.729  # Constriction factor
        else:
            self.inertia_weight = None  # Not used in standard PSO

    def optimize(self):
        for iteration in range(self.max_iter):
            for i in range(self.population_size):
                fitness = self.fitness_function(self.positions[i])
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i]

            self.loss_history.append(self.global_best_score)

            for i in range(self.population_size):
                if self.variant == "inertia":
                    self.inertia_weight = max(0.4, self.inertia_weight - 0.005)  # Decay inertia
                    inertia = self.inertia_weight * self.velocities[i]
                elif self.variant == "constricted":
                    inertia = self.chi * self.velocities[i]
                else:
                    inertia = self.alpha * self.velocities[i]

                cognitive = self.beta * np.random.uniform() * (self.personal_best_positions[i] - self.positions[i])
                social = self.gamma * np.random.uniform() * (self.global_best_position - self.positions[i])
                self.velocities[i] = inertia + cognitive + social
                self.positions[i] += self.velocities[i]

        return self.global_best_position, self.global_best_score
