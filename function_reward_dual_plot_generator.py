import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from sympy import symbols, diff, tanh, sqrt, pi
# taken and modified from: https://chat.openai.com/share/77ca91bc-d877-4663-8da0-a7babb5fb309


# Defining the GELU function
x = symbols('x')
gelu_expr = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x**3)))
gelu_func = np.vectorize(lambda x: gelu_expr.evalf(subs={symbols('x'): x}))

# Define the derivative functions
def first_derivative(func, x0):
    return derivative(func, x0, dx=1e-6)

def second_derivative(func, x0):
    return derivative(func, x0, dx=1e-6, n=2)

# Define the curvature function
def curvature(y1, y2):
    return np.abs(y2) / (1 + y1**2)**1.5

# Range of x
x_values = np.linspace(-8, 8, 400)

# Calculate the values of the GELU function
y_values = gelu_func(x_values)

# Calculate the first and second derivatives
y1_values = np.array([first_derivative(gelu_func, x0) for x0 in x_values])
y2_values = np.array([second_derivative(gelu_func, x0) for x0 in x_values])

# Calculate the curvature
curvature_values = curvature(y1_values, y2_values)

# Define the points to be marked
marked_points = [-3, -2, -1, 0]

# Calculate the values of the GELU function and the curvature at the marked points
gelu_marked_values = gelu_func(marked_points)
curvature_marked_values = curvature(np.array([first_derivative(gelu_func, x0) for x0 in marked_points]),
                                    np.array([second_derivative(gelu_func, x0) for x0 in marked_points]))


### Final version to be saved

# Double the Size of text in plot
'''
current_size = plt.rcParams['axes.titlesize']

plt.rcParams['axes.titlesize'] = 2 * current_size
plt.rcParams['axes.labelsize'] = 2 * plt.rcParams['axes.labelsize']
plt.rcParams['legend.fontsize'] = 2 * plt.rcParams['legend.fontsize']
plt.rcParams['xtick.labelsize'] = 2 * plt.rcParams['xtick.labelsize']
plt.rcParams['ytick.labelsize'] = 2 * plt.rcParams['ytick.labelsize']
'''

# Mapping from relative size to absolute size in points
size_mapping = {
    'xx-small': 8,
    'x-small': 10,
    'small': 12,
    'medium': 14,
    'large': 16,
    'x-large': 20,
    'xx-large': 24,
}

# Convert the sizes to points, double them, and set them
plt.rcParams['axes.titlesize'] = 1.4 * size_mapping[plt.rcParams['axes.titlesize']]
plt.rcParams['axes.labelsize'] = 1.2 * size_mapping[plt.rcParams['axes.labelsize']]
plt.rcParams['legend.fontsize'] = 1.2 * size_mapping[plt.rcParams['legend.fontsize']]
plt.rcParams['xtick.labelsize'] = 1.2 * size_mapping[plt.rcParams['xtick.labelsize']]
plt.rcParams['ytick.labelsize'] = 1.2 * size_mapping[plt.rcParams['ytick.labelsize']]


# Create the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot the GELU function
ax1.plot(x_values, y_values, label='GELU', color='mediumseagreen')

# Scatter the marked points and annotate them with their coordinates, preserving the legend
for i, (x, y) in enumerate(zip(marked_points, gelu_marked_values)):
    color = 'red' if x != 0 else 'blue'
    marker = 'o' if x != 0 else 'x'
    label = 'Previous Points: (-3, -2, -1)' if i < 3 else 'New Point Chosen From Action=+1.0: (0)'
    ax1.scatter(x, y, c=color, marker=marker, label=label)
    ax1.text(x, float(y), f'({x}, {float(y):.2f})', fontsize=10)

# Add vertical lines
ax1.axvline(x=-1, color='green', linestyle='--', ymax=0.5)
ax1.axvline(x=0, color='purple', linestyle='--', ymax=0.5)

# Add arrow
ax1.annotate('', xy=(0, 0.5), xytext=(-1, 0.5), arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Add text at specified position with larger font size and bolded
ax1.text(-0.35, 1.5, 'action=+1.0\n=>new point: 0.0', ha='center', fontsize=12 * 1.5, weight='bold')

ax1.set_title('GELU Function')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Only add legend entries for the first occurrences
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys())

# Set x-axis limits to zoom in
ax1.set_xlim([-5, 5])

# Adjust y-axis limits to match the x-axis limits
y_min, y_max = ax1.get_ylim()
new_y_max = y_max * 4.5 / 6  # Adjust the upper y limit to match the upper x limit
ax1.set_ylim([y_min, new_y_max])

# Plot the curvature
ax2.plot(x_values, curvature_values, label='Curvature/Intermediate Reward', color='mediumseagreen')

# Scatter the marked points and annotate them with their coordinates
for x, y in zip(marked_points, curvature_marked_values):
    ax2.scatter(x, y, c='red' if x != 0 else 'blue', marker='o' if x != 0 else 'x')
    ax2.text(x, float(y), f'({x}, {float(y):.2f})', fontsize=10)

ax2.bar(marked_points[:-1], curvature_marked_values[:-1], color='orange', width=0.2, label='Previous Curvature/Reward: (-3, -2, -1)')
ax2.bar(marked_points[-1], curvature_marked_values[-1], color='blue', width=0.2, label='New Curvature/Reward: (0)')
ax2.set_title('Curvature/Intermediate Reward')
ax2.set_xlabel('x')
ax2.set_ylabel('Curvature/Intermediate Reward')
ax2.legend(loc='center')

# Set x-axis limits to zoom in
ax2.set_xlim([-5, 5])

plt.tight_layout()




plt.show()

fig.savefig('RL_function_reward_dual_plot_diagram_paper.png', bbox_inches='tight', dpi=300)

