import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from matplotlib.animation import FuncAnimation

# Generate synthetic data for predictions and true labels
np.random.seed(42)
y_true = np.random.randint(0, 2, 100)  # binary true labels
y_scores = np.random.rand(100)  # continuous scores between 0 and 1

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Set up the figure and axes
fig, (ax_roc, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

# Initial ROC curve
ax_roc.plot(fpr, tpr, color='blue', label='ROC Curve')
ax_roc.set_title('ROC Curve with Moving Threshold')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
roc_point, = ax_roc.plot([], [], 'ro')  # Initial red point on ROC curve

# Initial histogram
ax_hist.hist(y_scores[y_true == 0], bins=20, alpha=0.5, label='Class 0', color='blue')
ax_hist.hist(y_scores[y_true == 1], bins=20, alpha=0.5, label='Class 1', color='orange')
ax_hist.set_title('Score Distribution with Threshold')
ax_hist.set_xlabel('Score')
ax_hist.set_ylabel('Frequency')
threshold_line = ax_hist.axvline(x=0, color='red', linestyle='--', label='Threshold')
ax_hist.legend()


# Update function for animation
def update(i):
    # Update ROC point
    roc_point.set_data(fpr[i], tpr[i])

    # Update threshold line on histogram
    threshold_line.set_xdata([thresholds[i], thresholds[i]])

    return roc_point, threshold_line


# Create animation
ani = FuncAnimation(fig, update, frames=len(thresholds), blit=True, repeat=False)

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Generate sample data for demonstration
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x / 2)
y4 = np.exp(-x / 3) * np.sin(x * 2)

# Set up the figure with GridSpec
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(3, 2, width_ratios=[1, 3])  # Adjusting width ratios to make right plot larger

# Three plots on the left
ax1 = fig.add_subplot(gs[0, 0])  # Top-left plot
ax2 = fig.add_subplot(gs[1, 0])  # Middle-left plot
ax3 = fig.add_subplot(gs[2, 0])  # Bottom-left plot

# One plot on the right, spanning both rows
ax4 = fig.add_subplot(gs[:, 1])  # Right plot spanning both rows

# Plot data in each subplot
ax1.plot(x, y1, color='blue')
ax1.set_title('Plot 1: Sine')

ax2.plot(x, y2, color='green')
ax2.set_title('Plot 2: Cosine')

ax3.plot(x, y3, color='purple')
ax3.set_title('Plot 3: Tangent')

ax4.plot(x, y4, color='red')
ax4.set_title('Plot 4: Damped Sine Wave')

# Adjust layout and show plot
plt.tight_layout()
plt.show()
