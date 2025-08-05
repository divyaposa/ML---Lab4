# Import required libraries
import numpy as np  # For numerical operations like generating random floats
import matplotlib.pyplot as plt  # For plotting graphs and scatter plots
from typing import Tuple  # For returning multiple types from a function
from random import seed, randint  # For reproducible random label generation

# Class to represent a single 2D data point with a label (either class 0 or 1)
class DataPoint:
    def __init__(self, x: float, y: float, label: int):
        self.x = x  # X-coordinate (feature)
        self.y = y  # Y-coordinate (feature)
        self.label = label  # Class label: 0 for Blue, 1 for Red

    def __repr__(self):
        # Defines how each DataPoint object is printed (for easy viewing)
        return f"({self.x:.2f}, {self.y:.2f}) -> Class {self.label}"

# Class to generate synthetic labeled 2D data for classification
class SyntheticDataGenerator:
    def __init__(self, num_points: int = 20, min_val: int = 1, max_val: int = 10, seed_val: int = 42):
        """
        Constructor initializes:
        - num_points: number of data points to generate
        - min_val and max_val: range for feature values
        - seed_val: for reproducibility of randomness
        """
        self.num_points = num_points  # Total number of points to generate
        self.min_val = min_val  # Minimum value for features
        self.max_val = max_val  # Maximum value for features
        self.seed_val = seed_val  # Seed to ensure reproducibility
        self.data_points = []  # List to store generated DataPoint objects

    def generate_data(self) -> None:
        """
        Generates synthetic (x, y) points and assigns a random class label (0 or 1)
        """
        np.random.seed(self.seed_val)  # Set NumPy seed for reproducibility
        seed(self.seed_val)  # Set random module seed for label consistency

        for _ in range(self.num_points):  # Generate the required number of points
            x = np.random.uniform(self.min_val, self.max_val)  # Random float for X within range
            y = np.random.uniform(self.min_val, self.max_val)  # Random float for Y within range
            label = randint(0, 1)  # Randomly assign a class label: 0 or 1
            self.data_points.append(DataPoint(x, y, label))  # Create and store the DataPoint

    def get_data_by_class(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separates data points into two arrays by class label.
        Returns:
            class0: Points with label 0
            class1: Points with label 1
        """
        class0 = [(dp.x, dp.y) for dp in self.data_points if dp.label == 0]  # Filter and collect class 0 points
        class1 = [(dp.x, dp.y) for dp in self.data_points if dp.label == 1]  # Filter and collect class 1 points

        class0 = np.array(class0)  # Convert list of tuples to NumPy array
        class1 = np.array(class1)  # Same for class 1

        return class0, class1  # Return both class arrays

# Class to visualize the data points using a scatter plot
class DataPlotter:
    def __init__(self, class0: np.ndarray, class1: np.ndarray):
        self.class0 = class0  # Store class 0 data points
        self.class1 = class1  # Store class 1 data points

    def plot(self):
        """
        Plots the 2D data points using different colors for each class
        """
        plt.figure(figsize=(8, 6))  # Set the size of the plot

        # Plot class 0 points in blue
        if self.class0.size > 0:  # Only if class 0 data exists
            plt.scatter(self.class0[:, 0], self.class0[:, 1], color='blue', label='Class 0 (Blue)')

        # Plot class 1 points in red
        if self.class1.size > 0:  # Only if class 1 data exists
            plt.scatter(self.class1[:, 0], self.class1[:, 1], color='red', label='Class 1 (Red)')

        # Add axis labels and a title
        plt.xlabel("Feature X")  # Label for x-axis
        plt.ylabel("Feature Y")  # Label for y-axis
        plt.title("Scatter Plot of Synthetic Training Data")  # Plot title

        # Add legend and grid
        plt.legend()  # Show legend indicating class colors
        plt.grid(True)  # Show grid
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()  # Display the plot

# ========= Main Execution (Entry Point) ========= #
if __name__ == "__main__":
    # Step 1: Create an instance of the data generator
    generator = SyntheticDataGenerator()  # Use default 20 points, range 1–10

    # Step 2: Generate the synthetic labeled data
    generator.generate_data()  # Fill data_points list

    # Step 3: Print the generated data points to console
    print("Generated Data Points:")  # Title for printed output
    for point in generator.data_points:  # Loop through each data point
        print(point)  # Print the point using the __repr__ method

    # Step 4: Separate data into two groups based on their class
    class0_data, class1_data = generator.get_data_by_class()  # Get separate arrays

    # Step 5: Create an instance of the plotter and visualize the data
    plotter = DataPlotter(class0_data, class1_data)  # Pass class data to plotter
    plotter.plot()  # Display the scatter plot
