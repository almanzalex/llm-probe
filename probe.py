import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class LinearRegressionProbe:
    def __init__(self, input_dim, lr=0.01):
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn(1)
        self.lr = lr

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def compute_cost(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)

    def fit(self, X, y, training_steps=100):
        cost_history = []
        weight_history = []

        for step in range(training_steps):
            y_pred = self.predict(X)

            cost = self.compute_cost(y_pred, y)
            cost_history.append(cost)
            weight_history.append(self.weights.copy())

            n = X.shape[0]
            dW = (2 / n) * np.dot(X.T, (y_pred - y))
            db = (2 / n) * np.sum(y_pred - y)

            self.weights -= self.lr * dW
            self.bias -= self.lr * db

            print(f"Step {step + 1}/{training_steps}, Cost: {cost:.4f}")

        return cost_history, weight_history

def generate_synthetic_data(num_samples=100, embedding_dim=768):
    np.random.seed(42)
    X = np.random.randn(num_samples, embedding_dim)
    y = np.random.randint(0, 2, size=num_samples)
    return X, y

if __name__ == "__main__":
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    probe = LinearRegressionProbe(input_dim=X.shape[1], lr=0.01)

    print("Training the probe...")
    cost_history, weight_history = probe.fit(X_train, y_train, training_steps=50)

    y_pred = probe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test Mean Squared Error: {mse:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, label="Cost")
    plt.xlabel("Training Steps")
    plt.ylabel("Cost")
    plt.title("Cost History")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot([weights[0] for weights in weight_history], label="Weight[0]")
    plt.xlabel("Training Steps")
    plt.ylabel("Weight Value")
    plt.title("Weight History (First Weight)")
    plt.legend()
    plt.show()

    weight_matrix = np.array(weight_history)
    average_magnitudes = np.mean(np.abs(weight_matrix), axis=0)
    sorted_indices = np.argsort(average_magnitudes)[::-1]
    sorted_weight_matrix = weight_matrix[:, sorted_indices]

    plt.figure(figsize=(12, 8))
    plt.imshow(np.abs(sorted_weight_matrix), aspect="auto", cmap="viridis", norm=LogNorm(vmin=1e-2, vmax=np.abs(weight_matrix).max()))
    plt.colorbar(label="Weight Magnitude (Log Scale)")
    plt.xlabel("Embedding Dimensions (Sorted by Importance)")
    plt.ylabel("Training Steps")
    plt.title("Heatmap of Weight Magnitudes Over Training (Sorted Dimensions)")
    plt.show()