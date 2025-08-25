"""
A6. Repeat the exercises A3 to A5 for your project data considering any two features and classes.
"""
# ===================== Import Libraries =====================
import os   # To work with file paths and directories
import numpy as np   # For numerical operations (arrays, math)
import torch   # For deep learning operations
import torchvision.transforms as transforms   # For image preprocessing
from torchvision.datasets import ImageFolder   # To load dataset in folder structure
from torchvision.models import resnet18   # Pretrained ResNet18 model
from sklearn.decomposition import PCA   # For reducing features to 2D
from sklearn.neighbors import KNeighborsClassifier   # kNN algorithm
import matplotlib.pyplot as plt   # For plotting graphs
from tqdm import tqdm  # For showing progress bar during loops


# ========= Class 1: Load Data + Extract Features =========
class MiniImageNet2DProcessor:
    """
    This class loads 2 selected classes from miniImageNet dataset,
    extracts features using pre-trained ResNet18,
    and reduces features to 2 dimensions using PCA (for visualization).
    """

    def __init__(self, data_path, selected_classes: list, num_samples_per_class=10, device='cpu'):
        # Path to dataset
        self.data_path = data_path
        # List of 2 class names chosen
        self.selected_classes = selected_classes
        # Limit number of samples taken per class
        self.num_samples_per_class = num_samples_per_class
        # Choose device (CPU or GPU)
        self.device = torch.device(device)

        # Load ResNet18 model that is already trained on ImageNet
        self.model = resnet18(pretrained=True)
        # Remove last classification layer (we only need features, not class predictions)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        # Put model in evaluation mode (no training)
        self.model.eval().to(self.device)

        # To store features (X) and labels (y)
        self.X = []
        self.y = []
        # To store mapping of class index to class name
        self.class_to_idx = {}

    def _extract_features(self, loader):
        """
        Runs all selected images through ResNet18
        and extracts high-level features.
        """
        features = []  # Store feature vectors
        labels = []    # Store class labels

        # Disable gradient calculations (faster, no training happening)
        with torch.no_grad():
            for images, targets in tqdm(loader, desc=" Extracting features"):
                # Move batch of images to CPU/GPU
                images = images.to(self.device)
                # Pass images through ResNet (get features instead of predictions)
                output = self.model(images).squeeze(-1).squeeze(-1)
                # Convert tensor to numpy array and save
                features.append(output.cpu().numpy())
                # Save corresponding labels
                labels.extend(targets.numpy())

        # Stack all feature arrays vertically
        return np.vstack(features), np.array(labels)

    def load_and_process_data(self):
        """
        Loads dataset, selects only 2 classes,
        extracts features using ResNet,
        and reduces them to 2D using PCA.
        """
        # Step 1: Define preprocessing (resize and convert to tensor)
        transform = transforms.Compose([
            transforms.Resize((84, 84)),   # Resize all images to 84x84 pixels
            transforms.ToTensor()          # Convert images to PyTorch tensors
        ])

        # Step 2: Load dataset (must be in ImageFolder format: class_name/images)
        dataset = ImageFolder(self.data_path, transform=transform)

        # Step 3: Save mapping (index -> class name)
        self.class_to_idx = {v: k for k, v in dataset.class_to_idx.items()}

        # Step 4: Filter only the 2 classes we selected
        selected_indices = [idx for idx, (img_path, label) in enumerate(dataset.samples)
                            if dataset.classes[label] in self.selected_classes]

        # Step 5: Take only limited samples per class
        selected_data = {cls: [] for cls in self.selected_classes}
        for idx in selected_indices:
            path, label = dataset.samples[idx]
            cls = dataset.classes[label]
            if len(selected_data[cls]) < self.num_samples_per_class:
                selected_data[cls].append((path, label))

        # Step 6: Flatten dictionary into list of image samples
        new_samples = [item for sublist in selected_data.values() for item in sublist]
        dataset.samples = new_samples
        dataset.targets = [label for _, label in new_samples]

        # Step 7: Load data into batches
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        # Step 8: Extract deep features from ResNet
        self.X, self.y = self._extract_features(loader)

        # Step 9: Reduce feature size to 2D using PCA
        pca = PCA(n_components=2)
        self.X = pca.fit_transform(self.X)

        return self.X, self.y


# ========= Class 2: Visualization =========
class KNNVisualizer:
    """
    Handles plotting of training data points (after PCA)
    and decision boundaries of kNN classifier.
    """

    def __init__(self, X, y, class_names):
        self.X = X   # 2D PCA features
        self.y = y   # Labels
        self.class_names = class_names  # The 2 class names

    def plot_training_data(self):
        """
        Scatter plot of the training data in 2D space.
        """
        plt.figure(figsize=(8, 6))
        for label in np.unique(self.y):  # Loop through both classes
            plt.scatter(
                self.X[self.y == label, 0],  # X-axis: PCA feature 1
                self.X[self.y == label, 1],  # Y-axis: PCA feature 2
                label=self.class_names[label],  # Show class name in legend
                c='blue' if label == 0 else 'red',  # Color for classes
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
        Train kNN classifier with given k,
        predict for all points in grid,
        and plot decision boundary.
        """
        # Step 1: Train kNN classifier
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(self.X, self.y)

        # Step 2: Create a grid that covers plot area
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        test_points = np.c_[xx.ravel(), yy.ravel()]  # Flatten grid

        # Step 3: Predict for all grid points
        Z = clf.predict(test_points).reshape(xx.shape)

        # Step 4: Plot decision regions and training points
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
    # Path to dataset folder
    DATA_PATH = r'C:/Users/Divya/Desktop/labdataset'

    # Choose any 2 class folder names from dataset
    SELECTED_CLASSES = ['n01532829', 'n01749939']

    # Step 1: Load and process data
    processor = MiniImageNet2DProcessor(DATA_PATH, SELECTED_CLASSES)
    X, y = processor.load_and_process_data()

    # Step 2: Create visualizer object
    visualizer = KNNVisualizer(X, y, SELECTED_CLASSES)

    # A3: Plot raw training points (after PCA)
    visualizer.plot_training_data()

    # A4: Plot decision boundary for k = 3
    visualizer.plot_decision_boundary(k=3)

    # A5: Repeat decision boundary for multiple values of k
    for k in [1, 3, 5, 7, 11]:
        visualizer.plot_decision_boundary(k)
