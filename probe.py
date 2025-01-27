import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class LinearRegressionProbe:
    def __init__(self, input_dim, num_classes=2, lr=0.01):
        self.weights = np.random.randn(input_dim, num_classes)  # Multiclass weights
        self.bias = np.random.randn(num_classes)  # Multiclass bias
        self.lr = lr

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Stability trick
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        return self.softmax(logits)

    def compute_cost(self, y_pred, y):
        n = y.shape[0]
        log_likelihood = -np.log(y_pred[np.arange(n), y])
        return np.mean(log_likelihood)

    def fit(self, X, y, training_steps=100):
        cost_history = []
        for step in range(training_steps):
            logits = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(logits)

            cost = self.compute_cost(y_pred, y)
            cost_history.append(cost)

            # Gradient computation
            n = X.shape[0]
            y_one_hot = np.zeros_like(y_pred)
            y_one_hot[np.arange(n), y] = 1
            dW = np.dot(X.T, (y_pred - y_one_hot)) / n
            db = np.sum(y_pred - y_one_hot, axis=0) / n

            # Update weights and bias
            self.weights -= self.lr * dW
            self.bias -= self.lr * db

            print(f"Step {step + 1}/{training_steps}, Cost: {cost:.4f}")

        return cost_history

# Load embeddings and labels from the JSON file
def load_embeddings(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    embeddings = np.array([entry["embedding"] for entry in data])
    labels = np.array([entry["label"] for entry in data], dtype=int)  # Ensure labels are integers
    return embeddings, labels

if __name__ == "__main__":
    # Load real embeddings and labels
    X, y = load_embeddings("gum_embeddings.json")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the probe
    probe = LinearRegressionProbe(input_dim=X.shape[1], num_classes=3, lr=0.001)
    print("Training the probe")
    cost_history = probe.fit(X_train, y_train, training_steps=50)

    # Evaluate the probe
    y_pred_probs = probe.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Visualize cost history
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, label="Cost")
    plt.xlabel("Training Steps")
    plt.ylabel("Cost")
    plt.title("Cost History")
    plt.legend()
    plt.show()
