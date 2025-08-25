
"""
A4. Generate test set data with values of X & Y varying between 0 and 10 with increments of 0.1.
This creates a test set of about 10,000 points. Classify these points with above training data using
kNN classifier (k = 3). Make a scatter plot of the test data output with test points colored as per their
predicted class colors (all points predicted class0 are labeled blue color). Observe the color spread
and class boundary lines in the feature space.
"""
# ========== Fixed KNN Test Data Visualization ========== #
# Import necessary libraries
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.neighbors import KNeighborsClassifier  # kNN classifier
from random import seed, randint  # For reproducible random numbers
from typing import Tuple, List  # For type hints

# Class representing a 2D data point with a class label
class DataPoint:
    def __init__(self, x: float, y: float, label: int):
        self.x = x  # X-coordinate
        self.y = y  # Y-coordinate
        self.label = label  # Class label (0 or 1)
    def to_features(self):
        return [self.x, self.y]  # Convert point to a feature list for kNN

# Class to generate random training data points
class SyntheticDataGenerator:
    def __init__(self, num_points=20, min_val=1, max_val=10, seed_val=42):
        self.num_points = num_points  # Total points to generate
        self.min_val = min_val  # Minimum feature value
        self.max_val = max_val  # Maximum feature value
        self.seed_val = seed_val  # Seed for reproducibility
        self.data_points: List[DataPoint] = []  # List to store data points

    def generate_training_data(self):
        np.random.seed(self.seed_val)  # Seed NumPy random generator
        seed(self.seed_val)  # Seed Python random generator
        for _ in range(self.num_points):  # Loop to generate each point
            x = np.random.uniform(self.min_val, self.max_val)  # Random X value
            y = np.random.uniform(self.min_val, self.max_val)  # Random Y value
            label = randint(0, 1)  # Random class 0 or 1
            self.data_points.append(DataPoint(x, y, label))  # Store the point

    def get_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array([dp.to_features() for dp in self.data_points])  # Feature array
        y = np.array([dp.label for dp in self.data_points])  # Label array
        return X, y  # Return features and labels

# Class to generate dense test data grid
class TestDataGenerator:
    def __init__(self, min_val=0, max_val=10, step=0.1):
        self.min_val = min_val  # Minimum X/Y value
        self.max_val = max_val  # Maximum X/Y value
        self.step = step  # Step size for grid

    def generate_test_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_vals = np.arange(self.min_val, self.max_val + self.step, self.step)  # X grid values
        y_vals = np.arange(self.min_val, self.max_val + self.step, self.step)  # Y grid values
        xx, yy = np.meshgrid(x_vals, y_vals)  # Create meshgrid for 2D plotting
        grid_points = np.c_[xx.ravel(), yy.ravel()]  # Flatten grid for prediction
        return grid_points, xx, yy  # Return points and grid arrays

# Class to visualize kNN predictions
class KNNVisualizer:
    def __init__(self, knn_model, test_points, xx, yy, X_train=None, y_train=None):
        self.knn_model = knn_model  # Trained kNN model
        self.test_points = test_points  # Test points to classify
        self.xx = xx  # X meshgrid
        self.yy = yy  # Y meshgrid
        self.X_train = X_train  # Training features (optional)
        self.y_train = y_train  # Training labels (optional)

    def predict_and_plot(self):
        print("Classifying test data using kNN (k=3)...")  # Inform user
        predictions = self.knn_model.predict(self.test_points)  # Predict classes
        zz = predictions.reshape(self.xx.shape)  # Reshape to grid shape for plotting

        plt.figure(figsize=(10, 8))  # Set figure size

        # Plot decision boundaries as colored regions
        plt.contourf(self.xx, self.yy, zz, cmap=plt.cm.RdBu, alpha=0.5)

        # Plot training points on top for reference
        if self.X_train is not None and self.y_train is not None:
            plt.scatter(self.X_train[self.y_train == 0, 0], self.X_train[self.y_train == 0, 1],
                        c='blue', edgecolor='k', label='Class 0 (Train)')  # Class 0 points
            plt.scatter(self.X_train[self.y_train == 1, 0], self.X_train[self.y_train == 1, 1],
                        c='red', edgecolor='k', label='Class 1 (Train)')  # Class 1 points

        plt.xlabel("Feature X")  # Label X-axis
        plt.ylabel("Feature Y")  # Label Y-axis
        plt.title("kNN (k=3) Test Data Classification")  # Plot title
        plt.legend()  # Show legend
        plt.grid(True)  # Show grid
        plt.tight_layout()  # Adjust layout
        plt.show()  # Display plot

# ===== Main program ===== #
if __name__ == "__main__":
    training_data_gen = SyntheticDataGenerator()  # Create generator
    training_data_gen.generate_training_data()  # Generate 20 training points
    X_train, y_train = training_data_gen.get_features_and_labels()  # Get arrays

    knn = KNeighborsClassifier(n_neighbors=3)  # Create kNN with k=3
    knn.fit(X_train, y_train)  # Train kNN model

    test_data_gen = TestDataGenerator()  # Create test data generator
    test_points, xx, yy = test_data_gen.generate_test_grid()  # Generate test grid

    visualizer = KNNVisualizer(knn, test_points, xx, yy, X_train, y_train)  # Create visualizer
    visualizer.predict_and_plot()  # Predict and plot results
