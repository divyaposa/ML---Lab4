"""
A5. Repeat A4 exercise for various values of k and observe the change in the class boundary lines.
"""

# Import necessary libraries
import numpy as np  # For numerical operations and arrays
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.neighbors import KNeighborsClassifier  # For k-Nearest Neighbors classifier
from random import seed, randint  # For generating reproducible random numbers
from typing import Tuple, List  # For type hints (clarity on return values and lists)


# Represents a single point in 2D space with a class label (0 or 1)
class DataPoint:
    def __init__(self, x: float, y: float, label: int):
        self.x = x                # X-coordinate
        self.y = y                # Y-coordinate
        self.label = label        # Class label (0 or 1)

    def to_features(self):
        # Converts the point to a list of features → [x, y]
        return [self.x, self.y]


# Generates synthetic training data with random coordinates and labels
class SyntheticDataGenerator:
    def __init__(self, num_points: int = 20, min_val: int = 1, max_val: int = 10, seed_val: int = 42):
        """
        num_points: total training samples to generate
        min_val, max_val: range for X and Y values
        seed_val: seed for reproducibility (same random data every run)
        """
        self.num_points = num_points
        self.min_val = min_val
        self.max_val = max_val
        self.seed_val = seed_val
        self.data_points: List[DataPoint] = []  # List to hold generated points

    def generate_training_data(self):
        # Set both numpy and random seeds for reproducibility
        np.random.seed(self.seed_val)
        seed(self.seed_val)

        # Generate random points and assign random class (0 or 1)
        for _ in range(self.num_points):
            x = np.random.uniform(self.min_val, self.max_val)  # Random float for X
            y = np.random.uniform(self.min_val, self.max_val)  # Random float for Y
            label = randint(0, 1)  # Randomly assign class 0 or 1
            self.data_points.append(DataPoint(x, y, label))  # Store in list

    def get_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts list of DataPoint objects into NumPy arrays:
        X → feature array shape (n_samples, 2)
        y → label array shape (n_samples,)
        """
        X = np.array([point.to_features() for point in self.data_points])  # [[x1,y1],[x2,y2],...]
        y = np.array([point.label for point in self.data_points])          # [0,1,0,1,...]
        return X, y


# Generates a 2D grid of test data points to visualize decision boundaries
class TestDataGenerator:
    def __init__(self, min_val: float = 0, max_val: float = 10, step: float = 0.1):
        """
        min_val, max_val: defines the grid boundaries
        step: resolution of grid (smaller step = more points, smoother boundary)
        """
        self.min_val = min_val
        self.max_val = max_val
        self.step = step

    def generate_test_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates a meshgrid (xx, yy) and flattens into list of (x, y) test points
        Returns:
        - grid_points: shape (n_points, 2)
        - xx, yy: 2D meshgrid arrays for plotting
        """
        x_vals = np.arange(self.min_val, self.max_val + self.step, self.step)  # X values
        y_vals = np.arange(self.min_val, self.max_val + self.step, self.step)  # Y values
        xx, yy = np.meshgrid(x_vals, y_vals)                                  # Create grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]                           # Flatten into list
        return grid_points, xx, yy


# Handles model training, prediction, and visualization for different values of k
class KNNVisualizer:
    def __init__(self, test_points: np.ndarray, xx: np.ndarray, yy: np.ndarray,
                 X_train: np.ndarray, y_train: np.ndarray):
        self.test_points = test_points  # All test grid points
        self.xx = xx                    # Meshgrid X for plotting
        self.yy = yy                    # Meshgrid Y for plotting
        self.X_train = X_train          # Training features
        self.y_train = y_train          # Training labels

    def predict_and_plot(self, k: int):
        """
        Trains a kNN model with given k value, predicts test grid,
        and plots decision boundaries + training points
        """
        # Step 1: Train KNN classifier
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(self.X_train, self.y_train)

        # Step 2: Predict labels for all test grid points
        predictions = knn_model.predict(self.test_points)

        # Step 3: Reshape predictions back to meshgrid shape (for contour plot)
        zz = predictions.reshape(self.xx.shape)

        # Step 4: Plot decision regions (background color) + training points
        plt.figure(figsize=(8, 6))
        plt.contourf(self.xx, self.yy, zz, cmap=plt.cm.RdBu, alpha=0.5)  # Decision regions
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1],              # Training points
                    c=self.y_train, cmap=plt.cm.RdBu, edgecolor='k', s=80, label='Train Data')

        # Step 5: Customize plot
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

    # Step 2: Convert training data into NumPy arrays (X, y)
    X_train, y_train = training_data_gen.get_features_and_labels()

    # Step 3: Generate test grid (dense 2D space to visualize decision boundaries)
    test_data_gen = TestDataGenerator()
    test_points, xx, yy = test_data_gen.generate_test_grid()

    # Step 4: Initialize visualizer with training and test data
    visualizer = KNNVisualizer(test_points, xx, yy, X_train, y_train)

    # Step 5: Loop through different k values and plot decision boundaries
    k_values = [1, 3, 5, 7, 11]  # Different neighbor values
    for k in k_values:
        print(f"\n Generating plot for k = {k}")  # Print status
        visualizer.predict_and_plot(k)            # Plot for current k
