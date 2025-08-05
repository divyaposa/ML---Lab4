# Import necessary libraries
import numpy as np  # For numerical operations and arrays
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.neighbors import KNeighborsClassifier  # For k-Nearest Neighbors classifier
from random import seed, randint  # For generating reproducible random numbers
from typing import Tuple, List  # For specifying return types of functions and list types


# Represents a single point in 2D space with a class label (0 or 1)
class DataPoint:
    def __init__(self, x: float, y: float, label: int):
        self.x = x
        self.y = y
        self.label = label

    def to_features(self):
        # Converts the point to a list of features (used by the model)
        return [self.x, self.y]


# Generates synthetic training data with random coordinates and labels
class SyntheticDataGenerator:
    def __init__(self, num_points: int = 20, min_val: int = 1, max_val: int = 10, seed_val: int = 42):
        """
        num_points: total training samples to generate
        min_val, max_val: range for X and Y values
        seed_val: for reproducibility (same data every time)
        """
        self.num_points = num_points
        self.min_val = min_val
        self.max_val = max_val
        self.seed_val = seed_val
        self.data_points: List[DataPoint] = []

    def generate_training_data(self):
        # Set seeds for consistent random results
        np.random.seed(self.seed_val)
        seed(self.seed_val)

        # Generate random points and assign random class (0 or 1)
        for _ in range(self.num_points):
            x = np.random.uniform(self.min_val, self.max_val)
            y = np.random.uniform(self.min_val, self.max_val)
            label = randint(0, 1)
            self.data_points.append(DataPoint(x, y, label))

    def get_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts list of DataPoint objects into NumPy arrays for training:
        X: feature array [[x1, y1], [x2, y2], ...]
        y: label array [0, 1, 0, 1, ...]
        """
        X = np.array([point.to_features() for point in self.data_points])
        y = np.array([point.label for point in self.data_points])
        return X, y


# Generates a 2D grid of test data points to visualize decision boundaries
class TestDataGenerator:
    def __init__(self, min_val: float = 0, max_val: float = 10, step: float = 0.1):
        """
        min_val, max_val: range for the grid
        step: how densely spaced the test points should be (smaller = more points)
        """
        self.min_val = min_val
        self.max_val = max_val
        self.step = step

    def generate_test_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates a meshgrid of X and Y coordinates.
        Returns:
        - grid_points: flattened array of all test (x, y) points
        - xx, yy: 2D arrays used for plotting
        """
        x_vals = np.arange(self.min_val, self.max_val + self.step, self.step)
        y_vals = np.arange(self.min_val, self.max_val + self.step, self.step)
        xx, yy = np.meshgrid(x_vals, y_vals)
        grid_points = np.c_[xx.ravel(), yy.ravel()]  # Flatten meshgrid to 2D point list
        return grid_points, xx, yy


# Handles model training, prediction, and visualization for different values of k
class KNNVisualizer:
    def __init__(self, test_points: np.ndarray, xx: np.ndarray, yy: np.ndarray, X_train: np.ndarray, y_train: np.ndarray):
        self.test_points = test_points
        self.xx = xx
        self.yy = yy
        self.X_train = X_train
        self.y_train = y_train

    def predict_and_plot(self, k: int):
        """
        Trains a kNN model with given k value, predicts test grid, and plots decision boundaries.
        """
        # Train kNN classifier
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(self.X_train, self.y_train)

        # Predict labels for all test grid points
        predictions = knn_model.predict(self.test_points)
        zz = predictions.reshape(self.xx.shape)  # Reshape to match meshgrid shape

        # Plotting decision boundaries and training points
        plt.figure(figsize=(8, 6))
        plt.contourf(self.xx, self.yy, zz, cmap=plt.cm.RdBu, alpha=0.5)  # Background color regions
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1],
                    c=self.y_train, cmap=plt.cm.RdBu, edgecolor='k', s=80, label='Train Data')  # Training data

        plt.title(f"Decision Boundary with k = {k}")
        plt.xlabel("Feature X")
        plt.ylabel("Feature Y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ========== Main Program Execution ========== #
if __name__ == "__main__":
    # Step 1: Generate 20 random training points with labels
    training_data_gen = SyntheticDataGenerator()
    training_data_gen.generate_training_data()

    # Step 2: Convert training data to model-friendly NumPy arrays
    X_train, y_train = training_data_gen.get_features_and_labels()

    # Step 3: Generate test grid (10,000+ points covering the 2D space)
    test_data_gen = TestDataGenerator()
    test_points, xx, yy = test_data_gen.generate_test_grid()

    # Step 4: Create visualizer object with training and test data
    visualizer = KNNVisualizer(test_points, xx, yy, X_train, y_train)

    # Step 5: Loop through different k values and plot decision boundaries
    k_values = [1, 3, 5, 7, 11]
    for k in k_values:
        print(f"\n🔍 Generating plot for k = {k}")
        visualizer.predict_and_plot(k)