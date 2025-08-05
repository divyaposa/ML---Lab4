# Importing necessary libraries
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar for long-running loops

# ========= MiniImageNet Preprocessing and Feature Extraction =========
class MiniImageNet2DProcessor:
    """
    Loads 2 specific classes from miniImageNet, extracts deep features using pre-trained ResNet18,
    and reduces them to 2D using PCA for visualization and classification.
    """
    def __init__(self, data_path, selected_classes: list, num_samples_per_class=10, device='cpu'):
        self.data_path = data_path
        self.selected_classes = selected_classes
        self.num_samples_per_class = num_samples_per_class
        self.device = torch.device(device)

        # Load pre-trained ResNet18 model and remove the classifier layer
        self.model = resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove final FC layer
        self.model.eval().to(self.device)  # Set to evaluation mode

        self.X = []  # Features
        self.y = []  # Labels
        self.class_to_idx = {}

    def _extract_features(self, loader):
        """
        Runs all images through ResNet18 to extract high-level features.
        """
        features = []
        labels = []

        with torch.no_grad():  # Disable gradient calculations
            for images, targets in tqdm(loader, desc="🔍 Extracting features"):
                images = images.to(self.device)
                output = self.model(images).squeeze(-1).squeeze(-1)  # Remove extra dimensions
                features.append(output.cpu().numpy())  # Convert to numpy
                labels.extend(targets.numpy())  # Collect labels

        return np.vstack(features), np.array(labels)

    def load_and_process_data(self):
        """
        Loads data from ImageFolder, filters only selected classes, and extracts PCA-reduced features.
        """
        # Define transformation: resize image and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])

        # Load dataset from folders
        dataset = ImageFolder(self.data_path, transform=transform)
        self.class_to_idx = {v: k for k, v in dataset.class_to_idx.items()}  # Mapping index to class name

        # Filter dataset: only include selected classes
        selected_indices = [idx for idx, (img_path, label) in enumerate(dataset.samples)
                            if dataset.classes[label] in self.selected_classes]

        # Take limited number of samples per class (e.g., 10 each)
        selected_data = {cls: [] for cls in self.selected_classes}
        for idx in selected_indices:
            path, label = dataset.samples[idx]
            cls = dataset.classes[label]
            if len(selected_data[cls]) < self.num_samples_per_class:
                selected_data[cls].append((path, label))

        # Flatten dictionary into list and replace dataset samples
        new_samples = [item for sublist in selected_data.values() for item in sublist]
        dataset.samples = new_samples
        dataset.targets = [label for _, label in new_samples]

        # Load data in batches using DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        # Extract deep features using ResNet
        self.X, self.y = self._extract_features(loader)

        # Reduce dimensionality to 2D for visualization
        pca = PCA(n_components=2)
        self.X = pca.fit_transform(self.X)

        return self.X, self.y


# ========= Visualization Class =========
class KNNVisualizer:
    """
    Handles plotting the PCA features and kNN decision boundaries.
    """
    def __init__(self, X, y, class_names):
        self.X = X  # 2D features
        self.y = y  # Class labels
        self.class_names = class_names  # List of class names (2 classes)

    def plot_training_data(self):
        """
        Scatter plot showing the PCA-reduced training data points.
        """
        plt.figure(figsize=(8, 6))
        for label in np.unique(self.y):
            plt.scatter(
                self.X[self.y == label, 0],  # x-axis: feature 1
                self.X[self.y == label, 1],  # y-axis: feature 2
                label=self.class_names[label],  # Label name
                c='blue' if label == 0 else 'red',
                edgecolor='k'
            )
        plt.title("A3: 2D Feature Scatter Plot of 2 miniImageNet Classes")
        plt.xlabel("PCA Feature 1")
        plt.ylabel("PCA Feature 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_decision_boundary(self, k):
        """
        Trains a kNN classifier with given `k`, predicts over a grid, and plots the decision boundary.
        """
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(self.X, self.y)

        # Create a mesh grid of points covering the plot area
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        test_points = np.c_[xx.ravel(), yy.ravel()]

        # Predict class for each point in the grid
        Z = clf.predict(test_points).reshape(xx.shape)

        # Plot the decision regions and training data
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)
        for label in np.unique(self.y):
            plt.scatter(
                self.X[self.y == label, 0],
                self.X[self.y == label, 1],
                label=f"{self.class_names[label]}",
                edgecolor='k',
                c='blue' if label == 0 else 'red'
            )
        plt.title(f"A4/A5: Decision Boundary (k = {k})")
        plt.xlabel("PCA Feature 1")
        plt.ylabel("PCA Feature 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# =================== MAIN =================== #
if __name__ == "__main__":
    # 🔧 Set your own dataset path and the two classes to compare
    DATA_PATH = r'C:/Users/Divya/Desktop/labdataset'
    SELECTED_CLASSES = ['n01532829', 'n01749939']  # Replace with any 2 class folder names

    # Step 1: Load and process data
    processor = MiniImageNet2DProcessor(DATA_PATH, SELECTED_CLASSES)
    X, y = processor.load_and_process_data()

    # Step 2: Create visualizer with the 2D features
    visualizer = KNNVisualizer(X, y, SELECTED_CLASSES)

    # A3: Visualize raw training points (after PCA)
    visualizer.plot_training_data()

    # A4: Visualize decision boundary for k = 3
    visualizer.plot_decision_boundary(k=3)

    # A5: Repeat decision boundary for different values of k
    for k in [1, 3, 5, 7, 11]:
        visualizer.plot_decision_boundary(k)