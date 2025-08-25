"""A3. Generate 20 data points (training set data) consisting of 2 features (X & Y) whose values vary
randomly between 1 & 10. Based on the values, assign these 20 points to 2 different classes (class0 -
Blue & class1 – Red). Make a scatter plot of the training data and color the points as per their class
color. Observe the plot.
"""
# Import required libraries
import numpy as np  # For math operations and generating random numbers
import matplotlib.pyplot as plt  # For making charts/plots
from typing import Tuple  # Helps specify that a function returns multiple things
from random import seed, randint  # For random numbers and reproducibility

# Class to store one data point with x, y, and its class label
class DataPoint:
    def __init__(self, x: float, y: float, label: int):
        self.x = x  # Store the X value (first feature)
        self.y = y  # Store the Y value (second feature)
        self.label = label  # Store the class (0 = Blue, 1 = Red)

    def __repr__(self):
        # How this object looks when printed
        return f"({self.x:.2f}, {self.y:.2f}) -> Class {self.label}"  # Show 2 decimals

# Class to create synthetic (fake) 2D data points
class SyntheticDataGenerator:
    def __init__(self, num_points: int = 20, min_val: int = 1, max_val: int = 10, seed_val: int = 42):
        self.num_points = num_points  # How many points we want
        self.min_val = min_val  # Minimum possible value for X or Y
        self.max_val = max_val  # Maximum possible value for X or Y
        self.seed_val = seed_val  # Seed to make random numbers same every time
        self.data_points = []  # Empty list to store all data points

    def generate_data(self) -> None:
        np.random.seed(self.seed_val)  # Set seed for numpy random numbers
        seed(self.seed_val)  # Set seed for random module (for labels)

        for _ in range(self.num_points):  # Loop for each point
            x = np.random.uniform(self.min_val, self.max_val)  # Random X between 1-10
            y = np.random.uniform(self.min_val, self.max_val)  # Random Y between 1-10
            label = randint(0, 1)  # Randomly choose class 0 or 1
            self.data_points.append(DataPoint(x, y, label))  # Add the point to the list

    def get_data_by_class(self) -> Tuple[np.ndarray, np.ndarray]:
        # Separate points by class
        class0 = [(dp.x, dp.y) for dp in self.data_points if dp.label == 0]  # Class 0 points
        class1 = [(dp.x, dp.y) for dp in self.data_points if dp.label == 1]  # Class 1 points

        class0 = np.array(class0)  # Convert list to numpy array for plotting
        class1 = np.array(class1)  # Same for class 1

        return class0, class1  # Give back both arrays

# Class to draw a scatter plot for the data
class DataPlotter:
    def __init__(self, class0: np.ndarray, class1: np.ndarray):
        self.class0 = class0  # Store class 0 points
        self.class1 = class1  # Store class 1 points

    def plot(self):
        plt.figure(figsize=(8, 6))  # Create a plot of size 8x6 inches

        if self.class0.size > 0:  # Only plot if class 0 has points
            plt.scatter(self.class0[:, 0], self.class0[:, 1], color='blue', label='Class 0 (Blue)')  # Blue dots

        if self.class1.size > 0:  # Only plot if class 1 has points
            plt.scatter(self.class1[:, 0], self.class1[:, 1], color='red', label='Class 1 (Red)')  # Red dots

        plt.xlabel("Feature X")  # Label X-axis
        plt.ylabel("Feature Y")  # Label Y-axis
        plt.title("Scatter Plot of Synthetic Training Data")  # Plot title

        plt.legend()  # Show which color is which class
        plt.grid(True)  # Show grid lines
        plt.tight_layout()  # Adjust spacing
        plt.show()  # Display the plot

# ===== Main code to run everything =====
if __name__ == "__main__":
    generator = SyntheticDataGenerator()  # Make the generator object
    generator.generate_data()  # Create 20 random points

    print("Generated Data Points:")  # Print heading
    for point in generator.data_points:  # Go through each point
        print(point)  # Print the point

    class0_data, class1_data = generator.get_data_by_class()  # Separate by class

    plotter = DataPlotter(class0_data, class1_data)  # Make plotter object
    plotter.plot()  # Draw the scatter plot
