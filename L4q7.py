"""
A7. Use RandomSearchCV() or GridSearchCV() operations to find the ideal ‘k’ value for your
kNN classifier. This is called hyper-parameter tuning.
"""

# Import required libraries
import numpy as np                     # For working with arrays and numerical data
import os                              # For file and folder path operations
import torch                           # PyTorch library for deep learning
from torchvision import transforms     # For image transformations (resize, tensor conversion)
from torchvision.datasets import ImageFolder   # To load images stored in folders
from torchvision.models import resnet18        # Pretrained ResNet18 CNN model
from sklearn.decomposition import PCA   # For dimensionality reduction
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # For hyperparameter tuning
from sklearn.neighbors import KNeighborsClassifier   # kNN model
from sklearn.model_selection import train_test_split  # (Not used, but for splitting data)
from tqdm import tqdm  # For showing progress bars

# ==========================================
#  Class to extract image features using ResNet18
# ==========================================
class FeatureExtractor:
    def __init__(self, device='cpu'):
        # Choose device (CPU or GPU)
        self.device = torch.device(device)
        
        # Load pre-trained ResNet18 model (trained on ImageNet)
        self.model = resnet18(pretrained=True)
        
        # Remove the final classification layer so only features are extracted
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        # Put the model in evaluation mode (no training, just inference)
        self.model.eval().to(self.device)

    def extract_features(self, dataloader):
        """
        Converts each image into a fixed-size feature vector using ResNet18
        """
        features = []   # To store extracted feature vectors
        labels = []     # To store labels of images

        # Disable gradient calculation (faster, uses less memory)
        with torch.no_grad():
            # Loop over all batches of images
            for imgs, targets in tqdm(dataloader, desc="Extracting Features"):
                imgs = imgs.to(self.device)          # Move images to CPU/GPU
                outputs = self.model(imgs).squeeze() # Extract features (shape: [batch_size, 512])
                features.append(outputs.cpu().numpy()) # Convert to NumPy and move to CPU
                labels.extend(targets.numpy())         # Save class labels

        # Stack features and return (X = features, y = labels)
        return np.vstack(features), np.array(labels)


# ==========================================
#  Class to load miniImageNet data (only selected classes)
# ==========================================
class MiniImageNetDataLoader:
    def __init__(self, data_dir, selected_classes, samples_per_class=20):
        self.data_dir = data_dir                   # Path to dataset
        self.selected_classes = selected_classes   # Which classes to load
        self.samples_per_class = samples_per_class # Limit number of samples per class

    def load_data(self):
        """
        Loads only selected classes from miniImageNet using ImageFolder.
        Limits number of samples per class.
        """
        # Apply transformations (resize all images + convert to tensor)
        transform = transforms.Compose([
            transforms.Resize((84, 84)),  # Resize image to 84x84 pixels
            transforms.ToTensor()         # Convert image to PyTorch tensor
        ])

        # Load dataset from folder (expects folder/classname/*.jpg format)
        dataset = ImageFolder(self.data_dir, transform=transform)

        # Map class names to numeric indices (e.g., 'dog'->0, 'cat'->1)
        class_to_idx = {cls: idx for idx, cls in enumerate(dataset.classes)}

        # Select only required classes and limit samples per class
        selected = {cls: [] for cls in self.selected_classes}
        for path, label in dataset.samples:
            cls = dataset.classes[label]   # Get class name from label
            # If this class is in selected list and limit not reached
            if cls in self.selected_classes and len(selected[cls]) < self.samples_per_class:
                selected[cls].append((path, label))
        
        # Flatten the list of selected samples
        final_samples = [item for sublist in selected.values() for item in sublist]
        dataset.samples = final_samples   # Update dataset with only required samples
        dataset.targets = [label for _, label in final_samples]  # Update targets

        # Create DataLoader for batch processing
        from torch.utils.data import DataLoader
        return DataLoader(dataset, batch_size=16, shuffle=False)


# ==========================================
#  Class to perform Hyperparameter Tuning for kNN
# ==========================================
class KNNHyperparameterTuner:
    def __init__(self, X, y):
        self.X = X   # Feature vectors
        self.y = y   # Labels

    def tune_k(self, method='grid'):
        """
        Tries different values of 'k' (number of neighbors) and finds the best one using cross-validation.
        method: 'grid' or 'random'
        """
        # Define possible values for k (neighbors) from 1 to 20
        param_grid = {'n_neighbors': list(range(1, 21))}
        knn = KNeighborsClassifier()  # Initialize kNN model

        # Choose tuning method: GridSearchCV or RandomizedSearchCV
        if method == 'grid':
            search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')  # Test all k values
        elif method == 'random':
            search = RandomizedSearchCV(knn, param_grid, cv=5, n_iter=10, scoring='accuracy')  # Test 10 random k values
        else:
            raise ValueError("method must be 'grid' or 'random'")

        # Print info about current method
        print(f"Running {method.capitalize()}SearchCV for kNN...")

        # Fit model on data (perform hyperparameter search)
        search.fit(self.X, self.y)

        # Print best results
        print(f" Best k: {search.best_params_['n_neighbors']}")
        print(f"Best Accuracy: {search.best_score_:.4f}")
        return search.best_params_['n_neighbors'], search.best_score_


# ==========================================
#  Main Execution (Pipeline)
# ==========================================
if __name__ == "__main__":
    # Path to miniImageNet dataset folder
    DATA_PATH = r'C:/Users/Divya/Desktop/labdataset'
    
    # Choose any two classes from dataset (replace with your class names)
    SELECTED_CLASSES = ['n01532829', 'n01749939']

    # Step 1: Load dataset (only selected classes, with 50 images per class)
    loader = MiniImageNetDataLoader(DATA_PATH, SELECTED_CLASSES, samples_per_class=50)
    dataloader = loader.load_data()

    # Step 2: Extract deep features using pretrained ResNet18
    extractor = FeatureExtractor(device='cuda' if torch.cuda.is_available() else 'cpu')
    X, y = extractor.extract_features(dataloader)

    # Step 3: Reduce feature dimensions to 2D using PCA (for simplicity / visualization)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Step 4: Tune 'k' for kNN using GridSearchCV (can also use RandomSearchCV)
    tuner = KNNHyperparameterTuner(X_2d, y)
    best_k, best_score = tuner.tune_k(method='grid')  # Try method='random' too!
