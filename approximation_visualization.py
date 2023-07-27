import numpy as np
import matplotlib.pyplot as plt


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# SiLU function
def silu(x):
    return x * sigmoid(x)

# GELU function
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# Piecewise linear approximation function
def piecewise_linear_approx(x, ranges, a, b):
    for i in range(len(ranges)):
        if ranges[i][0] <= x <= ranges[i][1]:
            return a[i] * x + b[i]
    return 0  # default case



def chart_function_and_approximations(function, function_name, ranges, a_list, b_list, save_file_name=None):
    # Plot function and its approximation with point marks for separating segments and coordinates
    range_minimum = ranges[0][0]
    range_maximum = ranges[-1][1]
    x = np.linspace(range_minimum, range_maximum, 1000)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
    fig, ax = plt.subplots(figsize=(12, 6))
    y = function(x)
    y_approx = [piecewise_linear_approx(i, ranges, a_list, b_list) for i in x]
    ax.plot(x, y, label=f"{function_name} function", linewidth=2, color='darkgray')
    for i in range(len(ranges)):
        x_piecewise = np.linspace(ranges[i][0], ranges[i][1], 500)
        y_piecewise = [piecewise_linear_approx(j, [ranges[i]], [a_list[i]], [b_list[i]]) for j in x_piecewise]
        ax.plot(x_piecewise, y_piecewise, label=f"{a_list[i]}x + {b_list[i]} for ({ranges[i][0]}, {ranges[i][1]}]", linestyle='dashed',
                color=colors[i])
        ax.plot(ranges[i][0],
                piecewise_linear_approx(ranges[i][0], [ranges[i]], [a_list[i]], [b_list[i]]), 'ko',
                markersize=4)  # lower bound
        ax.plot(ranges[i][1],
                piecewise_linear_approx(ranges[i][1], [ranges[i]], [a_list[i]], [b_list[i]]), 'ko',
                markersize=4)  # upper bound
        # adjust_text_position(ranges[i][0], piecewise_linear_approx(ranges[i][0], [ranges[i]], [a_list[i]], [b_list[i]]), f'({ranges[i][0]:.2f}, {piecewise_linear_approx(ranges[i][0], [ranges[i]], [a_list[i]], [b_list[i]]):.2f})', ax)  # lower bound coordinate
        # adjust_text_position(ranges[i][1], piecewise_linear_approx(ranges[i][1], [ranges[i]], [a_list[i]], [b_list[i]]), f'({ranges[i][1]:.2f}, {piecewise_linear_approx(ranges[i][1], [ranges[i]], [a_list[i]], [b_list[i]]):.2f})', ax)  # upper bound coordinate
    ax.set_title(f"{function_name} Function and Its Piecewise Linear Approximations")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    if save_file_name is not None:
        fig.savefig(save_file_name, bbox_inches='tight', dpi=300)


# Ranges, slopes, and intercepts for SiLU
ranges_silu = [[-8, -4.42526], [-4.42526, -1.26488], [-1.26488, -0.74592], [-0.74592, -0.16151], [-0.16151, 0.26769], [0.26769, 0.86067], [0.86067, 1.3285], [1.3285, 4.2054], [4.2054, 8]]
a_silu = [-0.01165, -0.07942, 0.07505, 0.28146, 0.526, 0.76872, 0.94894, 1.08379, 1.01433]
b_silu = [-0.0891, -0.389, -0.19362, -0.03966, -0.00016, -0.06514, -0.22024, -0.3994, -0.10728]

# Ranges, slopes, and intercepts for GELU
ranges_gelu = [[-8, -5.47388], [-5.47388, -2.51312], [-2.51312, -0.74613], [-0.74613, -0.37654], [-0.37654, -0.00782], [-0.00782, 0.35102], [0.35102, 0.71723], [0.71723, 2.581], [2.581, 8]]
a_gelu = [0.0004, -0.00131, -0.09836, 0.09909, 0.34544, 0.64001, 0.88308, 1.0946, 1.00022]
b_gelu = [0.00284, -0.00653, -0.25041, -0.10309, -0.01033, -0.00802, -0.09334, -0.24505, -0.00147]

# Ranges, slopes, and intercepts for Sigmoid
ranges_sigmoid = [[-8, -4.09369], [-4.09369, -2.66771], [-2.66771, -1.71237], [-1.71237, -0.91118], [-0.91118, 0.91171], [0.91171, 1.71749], [1.71749, 2.66826], [2.66826, 4.10567], [4.10567, 8]]
a_sigmoid = [0.00347, 0.034, 0.09213, 0.16718, 0.23986, 0.16683, 0.09199, 0.03379, 0.00343]
b_sigmoid = [0.02524, 0.1502, 0.30528, 0.43378, 0.50001, 0.56659, 0.69513, 0.85042, 0.97506]

# Sigmoid
chart_function_and_approximations(sigmoid, "Sigmoid", ranges_sigmoid, a_sigmoid, b_sigmoid, "sigmoid_approximation_visualized.png")
# SiLU
chart_function_and_approximations(silu, "SiLU", ranges_silu, a_silu, b_silu, "silu_approximation_visualized.png")
# GELU
chart_function_and_approximations(gelu, "GELU", ranges_gelu, a_gelu, b_gelu, "gelu_approximation_visualized.png")

plt.close()
# Plot function and its approximation with point marks for separating segments and coordinates combined into one plot
functions = [sigmoid, silu, gelu]
function_names = ["Sigmoid", "SiLU", "GELU"]
function_colors = ['red', 'green', 'blue']
marker_colors = ['ro', 'go', 'bo']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']

ranges = [ranges_sigmoid, ranges_silu, ranges_gelu]
a_lists = [a_sigmoid, a_silu, a_gelu]
b_lists = [b_sigmoid, b_silu, b_gelu]

# Make the texts in plot larger to be more readable
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


# Create the figure and axes

fig, ax = plt.subplots(figsize=(18, 9))

# Generate the data and plot the functions and their approximations
foo = 0
lines = []
for function, function_name, range_list, a_list, b_list in zip(functions, function_names, ranges, a_lists, b_lists):
    # Plot function and its approximation with point marks for separating segments and coordinates
    range_minimum = range_list[0][0]
    range_maximum = range_list[-1][1]
    x = np.linspace(range_minimum, range_maximum, 1000)

    y = function(x)
    y_approx = [piecewise_linear_approx(i, range_list, a_list, b_list) for i in x]

    # comma to unpack tuple
    line, = ax.plot(x, y, label=f"{function_name} function", linewidth=2, color=function_colors[foo])
    lines.append(line)

    for i in range(len(range_list)):
        x_piecewise = np.linspace(range_list[i][0], range_list[i][1], 500)
        y_piecewise = [piecewise_linear_approx(j, [range_list[i]], [a_list[i]], [b_list[i]]) for j in x_piecewise]

        # comma to unpack tuple
        line, = ax.plot(x_piecewise, y_piecewise, label=f"{a_list[i]}x + {b_list[i]} for ({range_list[i][0]}, {range_list[i][1]}]", linestyle='dashed',
                color=colors[i])
        lines.append(line)
        ax.plot(range_list[i][0], piecewise_linear_approx(range_list[i][0], [range_list[i]], [a_list[i]], [b_list[i]]),
                marker_colors[foo], markersize=4)  # lower bound
        ax.plot(range_list[i][1], piecewise_linear_approx(range_list[i][1], [range_list[i]], [a_list[i]], [b_list[i]]),
                marker_colors[foo], markersize=4)  # upper bound

    foo = foo + 1

# Set the title and labels, show the legend, and display the plot

print(len(lines))
# Separate the entries into two lists: first 20 entries and remaining 10 entries
lines1, lines2 = lines[:20], lines[20:]
labels1, labels2 = [line.get_label() for line in lines1], [line.get_label() for line in lines2]
# First legend with sigmoid and SiLU for first column
first_legend = ax.legend(lines1, labels1, ncol=1, loc='upper left', bbox_to_anchor=(0.0, 1.01))
# Second legend with GELU for second column
second_legend = ax.legend(lines2, labels2, ncol=1, loc='upper left', bbox_to_anchor=(0.36, 1.01))

# Add the first legend manually to the current Axes
ax.add_artist(first_legend)

ax.set_title("Functions and Their Piecewise Linear Approximations")
ax.set_xlabel("x")
ax.set_ylabel("y")
# ax.legend(loc='upper left', ncol=2)
ax.grid(True)
plt.tight_layout()
plt.show()
fig.savefig('Combined_approximation_visualization.png', bbox_inches='tight', dpi=300)