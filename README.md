# Optimizing Artificial Neural Network (ANN) with Particle Swarm Optimization (PSO) for Concrete Strength Prediction

## Project Overview
This project demonstrates the optimization of an Artificial Neural Network (ANN) using Particle Swarm Optimization (PSO) to accurately predict the compressive strength of concrete. By leveraging PSO, the model achieves improved performance by finding optimal hyperparameters like the number of neurons and learning rate. 

The dataset used includes key features like cement, water, and aggregate composition, and the target variable is the compressive strength of the concrete.

---

## Dataset
- **Name**: Concrete Compressive Strength Dataset
- **Source**: UCI Machine Learning Repository
- **Features**:
  - Cement (kg/m³)
  - Blast Furnace Slag (kg/m³)
  - Fly Ash (kg/m³)
  - Water (kg/m³)
  - Superplasticizer (kg/m³)
  - Coarse Aggregate (kg/m³)
  - Fine Aggregate (kg/m³)
  - Age (days)
- **Target**: Concrete compressive strength (MPa)

---

## Methodology
1. **Data Preprocessing**:
   - Normalized input features for consistent scaling.
   - Split the dataset into training and testing subsets.
   
2. **ANN Architecture**:
   - Input layer with 8 features.
   - 1 or more hidden layers with tunable neurons.
   - Output layer for regression (single node).

3. **Optimization with PSO**:
   - Swarm of particles initialized with random hyperparameters (e.g., number of neurons, learning rate).
   - Fitness function evaluates ANN performance on training data.
   - Particles iteratively update positions based on local and global optima.

4. **Evaluation**:
   - Metrics used: Mean Squared Error (MSE) and R² score on test data.
   - Results compared with a baseline ANN model without optimization.

---

## Results
- PSO significantly improved the performance of the ANN compared to manual tuning.
- The optimized ANN demonstrated lower MSE and higher accuracy in predicting concrete strength.
