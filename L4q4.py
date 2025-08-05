# Import required libraries
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting results
from sklearn.neighbors import KNeighborsClassifier  # kNN model
from random import seed, randint  # For reproducible random class labels
from typing import Tuple, List  # For type annotations


# Class representing a 2D data point with a label (0 or 1)
class DataPoint:
    def __init__(self, x: float, y: float, label: int):
        self.x = x  # Feature X
        self.y = y  # Feature Y
        self.label = label  # Class label (0 or 1)

    def to_features(self):
        # Returns the point as a feature vector [x, y]
        return [self.x, self.y]


# Class to generate synthetic 2D training data
class SyntheticDataGenerator:
    def __init__(self, num_points: int = 20, min_val: int = 1, max_val: int = 10, seed_val: int = 42):
        """
        Parameters:
        - num_points: How many data points to generate
        - min_val, max_val: Range of values for x and y
        - seed_val: Seed for reproducibility
        """
        self.num_points = num_points
        self.min_val = min_val
        self.max_val = max_val
        self.seed_val = seed_val
        self.data_points: List[DataPoint] = []

    def generate_training_data(self):
        # Set seeds for reproducibility
        np.random.seed(self.seed_val)
        seed(self.seed_val)

        # Generate random 2D points and assign random class labels (0 or 1)
        for _ in range(self.num_points):
            x = np.random.uniform(self.min_val, self.max_val)
            y = np.random.uniform(self.min_val, self.max_val)
            label = randint(0, 1)  # Randomly assign label 0 or 1
            self.data_points.append(DataPoint(x, y, label))

    def get_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        # Extract X (features) and y (labels) as NumPy arrays for model training
        X = np.array([point.to_features() for point in self.data_points])
        y = np.array([point.label for point in self.data_points])
        return X, y


# Class to generate a dense grid of test points across 2D space
class TestDataGenerator:
    def __init__(self, min_val: float = 0, max_val: float = 10, step: float = 0.1):
        self.min_val = min_val
        self.max_val = max_val
        self.step = step

    def generate_test_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Create grid of (x, y) points spaced by 'step'
        x_vals = np.arange(self.min_val, self.max_val + self.step, self.step)
        y_vals = np.arange(self.min_val, self.max_val + self.step, self.step)

        # Create meshgrid of coordinates
        xx, yy = np.meshgrid(x_vals, y_vals)

        # Flatten grid to list of test points
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        return grid_points, xx, yy


# Class to visualize prediction results of kNN on the test grid
class KNNVisualizer:
    def __init__(self, knn_model: KNeighborsClassifier, test_points: np.ndarray, xx: np.ndarray, yy: np.ndarray):
        self.knn_model = knn_model
        self.test_points = test_points
        self.xx = xx
        self.yy = yy

    def predict_and_plot(self):
        # Predict class labels for all test grid points
        print("🔍 Classifying test data using kNN (k=3)...")
        predictions = self.knn_model.predict(self.test_points)

        # Reshape predictions back to the meshgrid shape
        zz = predictions.reshape(self.xx.shape)

        # Plot the decision boundary using color map
        plt.figure(figsize=(10, 8))
        plt.contourf(self.xx, self.yy, zz, cmap=plt.cm.RdBu, alpha=0.5)
        plt.colorbar(label='Predicted Class')  # Add color scale

        # Plot settings
        plt.xlabel("Feature X")
        plt.ylabel("Feature Y")
        plt.title("Test Data Classification using kNN (k=3)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ========== Main Program Execution ========== #
if __name__ == "__main__":
    # Step 1: Generate training data (20 random points with labels)
    training_data_gen = SyntheticDataGenerator()
    training_data_gen.generate_training_data()

    # Step 2: Convert the training data to model-friendly format
    X_train, y_train = training_data_gen.get_features_and_labels()

    # Step 3: Create and train a kNN classifier (k=3)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Step 4: Generate test data points across 2D space (dense grid)
    test_data_gen = TestDataGenerator()
    test_points, xx, yy = test_data_gen.generate_test_grid()

    # Step 5: Use the trained model to predict and visualize results
    visualizer = KNNVisualizer(knn, test_points, xx, yy)
    visualizer.predict_and_plot()