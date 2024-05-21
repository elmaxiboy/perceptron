import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load dataset
dataset = np.load("synthetic_dataset.npz")
coordinates = dataset['X']
labels = dataset['y']


# Map labels to colors
column_colors = {-1: 'r', 1: 'g'}
colors = list(map(lambda x: column_colors[x], labels[:]))

# Initialize perceptron variables
i = 0
iter = 0
consecutive_no_update = 0
weights = np.random.rand(3)
eta = 0.2
max_iteration = 10000  # Adjust the maximum number of iterations as needed

# Create figure and axis
fig, ax = plt.subplots()
scat = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, s=60)
xlim = ax.get_xlim()

# Initialize the plot limits
ax.set_xlim(-1, 4)
ax.set_ylim(-1, 4)
ax.set_aspect("equal")
fig.tight_layout()

# Perceptron update function
def perceptron(point, label):
    global weights, eta, consecutive_no_update
    data_ext = np.ones(3)
    data_ext[:2] = point
    prod = np.dot(weights, data_ext)
    pred = np.sign(prod)
    delta = eta * (label - pred) * data_ext
    if label != pred:
        weights += delta
        consecutive_no_update = 0
    else:
        consecutive_no_update += 1

# Animation update function
def update(frame):
    global i, consecutive_no_update, iter
    if consecutive_no_update >= 100:
        plt.close()
        return
    perceptron(coordinates[i, :], labels[i])
    ax.clear()
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, s=60)
    if np.abs(weights[1]) > 0.0001:
        x_vals = np.linspace(xlim[0], xlim[1], 100)
        y_vals = -(x_vals * weights[0]) / weights[1] - weights[2] / weights[1]
        ax.plot(x_vals, y_vals, linewidth=2, color='b')
    else:
        ax.axvline(x=-(weights[2] / weights[1]), ymin=-0.5, ymax=7, linewidth=2, color='b')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_aspect("equal")
    ax.set_title(f"Iteration: {iter}, Weights: {weights}, Consecutive No Updates: {consecutive_no_update}")
    fig.tight_layout()
    
    i = (i + 1) % coordinates.shape[0]
    iter += 1
    return scat,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=np.arange(max_iteration), repeat=False, blit=False)

# Display the plot
plt.show()
