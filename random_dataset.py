import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate 50 points for each class with reduced standard deviation
n_samples_per_class = 50
std_dev = 0.7  # Reduced standard deviation for tighter clustering

# Class 1: centered around (1, 1)
class_1 = std_dev * np.random.randn(n_samples_per_class, 2) + [1, 1]
labels_1 = np.ones((n_samples_per_class, 1))

# Class -1: centered around (2, 2)
class_2 = std_dev * np.random.randn(n_samples_per_class, 2) + [2, 2]
labels_2 = -np.ones((n_samples_per_class, 1))

# Combine the points and labels
data_class_1 = np.hstack((class_1, labels_1))
data_class_2 = np.hstack((class_2, labels_2))

# Combine the dataset
dataset = np.vstack((data_class_1, data_class_2))

# Shuffle the dataset
np.random.shuffle(dataset)

# Separate features and labels
X = dataset[:, :2]
y = dataset[:, 2]

# Visualize the dataset
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='green', label='Class 1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.title('Synthetic Dataset with Two Classes (Closer Clusters)')
plt.show()

# Save dataset for use in perceptron training
np.savez('synthetic_dataset_closer_classes.npz', X=X, y=y)

# Print the dataset shape
print("Dataset shape:", dataset.shape)
print("First 5 rows of dataset:\n", dataset[:5])
