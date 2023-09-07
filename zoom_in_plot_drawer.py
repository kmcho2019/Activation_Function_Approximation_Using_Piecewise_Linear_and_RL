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

# Ranges, slopes, and intercepts for SiLU
ranges_silu = [[-8, -4.42526], [-4.42526, -1.26488], [-1.26488, -0.74592], [-0.74592, -0.16151], [-0.16151, 0.26769], [0.26769, 0.86067], [0.86067, 1.3285], [1.3285, 4.2054], [4.2054, 8]]
a_silu = [-0.01165, -0.07942, 0.07505, 0.28146, 0.526, 0.76872, 0.94894, 1.08379, 1.01433]
b_silu = [-0.0891, -0.389, -0.19362, -0.03966, -0.00016, -0.06514, -0.22024, -0.3994, -0.10728]

# Ranges, slopes, and intercepts for GELU
ranges_gelu = [[-8, -5.47388], [-5.47388, -2.51312], [-2.51312, -0.74613], [-0.74613, -0.37654], [-0.37654, -0.00782], [-0.00782, 0.35102], [0.35102, 0.71723], [0.71723, 2.581], [2.581, 8]]
a_gelu = [0.0004, -0.00131, -0.09836, 0.09909, 0.34544, 0.64001, 0.88308, 1.0946, 1.00022]
b_gelu = [0.00284, -0.00653, -0.25041, -0.10309, -0.01033, -0.00802, -0.09334, -0.24505, -0.00147]

# Ranges, slopes, and intercepts
ranges = [[-8, -4.09369], [-4.09369, -2.66771], [-2.66771, -1.71237], [-1.71237, -0.91118], [-0.91118, 0.91171], [0.91171, 1.71749], [1.71749, 2.66826], [2.66826, 4.10567], [4.10567, 8]]
a = [0.00347, 0.034, 0.09213, 0.16718, 0.23986, 0.16683, 0.09199, 0.03379, 0.00343]
b = [0.02524, 0.1502, 0.30528, 0.43378, 0.50001, 0.56659, 0.69513, 0.85042, 0.97506]

# Create x values
x = np.linspace(-8, 8, 1000)

# Create y values
y_sigmoid = sigmoid(x)
y_piecewise = [piecewise_linear_approx(i, ranges, a, b) for i in x]

# Plot sigmoid
plt.figure(figsize=(10, 6))
plt.plot(x, y_sigmoid, label="Sigmoid function")

# Plot piecewise approximation
plt.plot(x, y_piecewise, label="Piecewise linear approximation", linestyle='dashed')

plt.title("Sigmoid Function and Its Piecewise Linear Approximations")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
# Plot sigmoid with lighter line and gray color
plt.close()
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']

plt.figure(figsize=(12, 6))
plt.plot(x, y_sigmoid, label="Sigmoid function", linewidth=2, color='gray')

a_list = []
b_list = []

# Plot each piecewise approximation segment with different color and mark interval points
for i in range(len(ranges)):
    # Create x values for the current range
    x_piecewise = np.linspace(ranges[i][0], ranges[i][1], 500)

    # Create y values for the current range
    y_piecewise = [piecewise_linear_approx(j, [ranges[i]], [a[i]], [b[i]]) for j in x_piecewise]
    a_ = x_piecewise
    b_ = y_piecewise
    a_list.append(a_)
    b_list.append(b_)

    # Plot the current segment
    plt.plot(x_piecewise, y_piecewise, label=f"{a[i]}x + {b[i]} for {ranges[i]}", linestyle='dashed', color=colors[i])

# Draw vertical lines with identical color for the same x-coordinate
for i in range(len(ranges) - 1):  # Exclude the last range upper limit
    plt.axvline(x=ranges[i][1], linestyle='dotted', color=colors[i])

plt.title("Sigmoid Function and Its Piecewise Linear Approximations")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.grid(True)
plt.tight_layout()
plt.show()

plt.plot(a_list[-1], b_list[-1])
plt.show()

print(b_list[-1])


# Function to adjust text position for preventing overlap
def adjust_text_position(x, y, text, ax):
    if x > 0:
        ha = 'left'
        x_pos = x + 0.1
    else:
        ha = 'right'
        x_pos = x - 0.1

    if y > 0.5:
        va = 'bottom'
        y_pos = y + 0.05
    else:
        va = 'top'
        y_pos = y - 0.05

    ax.text(x_pos, y_pos, text, ha=ha, va=va)

# Plot SiLU and its approximation with point marks for separating segments and coordinates
fig, ax = plt.subplots(figsize=(12, 6))
y_silu = silu(x)
y_silu_approx = [piecewise_linear_approx(i, ranges_silu, a_silu, b_silu) for i in x]
ax.plot(x, y_silu, label="SiLU function", linewidth=2, color='lightgray')
for i in range(len(ranges_silu)):
    x_piecewise = np.linspace(ranges_silu[i][0], ranges_silu[i][1], 500)
    y_piecewise = [piecewise_linear_approx(j, [ranges_silu[i]], [a_silu[i]], [b_silu[i]]) for j in x_piecewise]
    ax.plot(x_piecewise, y_piecewise, label=f"{a_silu[i]}x + {b_silu[i]} for {ranges_silu[i]}", linestyle='dashed', color=colors[i])
    ax.plot(ranges_silu[i][0], piecewise_linear_approx(ranges_silu[i][0], [ranges_silu[i]], [a_silu[i]], [b_silu[i]]), 'ko', markersize=5)  # lower bound
    ax.plot(ranges_silu[i][1], piecewise_linear_approx(ranges_silu[i][1], [ranges_silu[i]], [a_silu[i]], [b_silu[i]]), 'ko', markersize=5)  # upper bound
    # adjust_text_position(ranges_silu[i][0], piecewise_linear_approx(ranges_silu[i][0], [ranges_silu[i]], [a_silu[i]], [b_silu[i]]), f'({ranges_silu[i][0]:.2f}, {piecewise_linear_approx(ranges_silu[i][0], [ranges_silu[i]], [a_silu[i]], [b_silu[i]]):.2f})', ax)  # lower bound coordinate
    # adjust_text_position(ranges_silu[i][1], piecewise_linear_approx(ranges_silu[i][1], [ranges_silu[i]], [a_silu[i]], [b_silu[i]]), f'({ranges_silu[i][1]:.2f}, {piecewise_linear_approx(ranges_silu[i][1], [ranges_silu[i]], [a_silu[i]], [b_silu[i]]):.2f})', ax)  # upper bound coordinate
ax.set_title("SiLU Function and Its Piecewise Linear Approximations")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
ax.grid(True)
plt.tight_layout()
plt.show()
# Plot GELU and its approximation with point marks for separating segments and coordinates
fig, ax = plt.subplots(figsize=(12, 6))
y_gelu = gelu(x)
y_gelu_approx = [piecewise_linear_approx(i, ranges_gelu, a_gelu, b_gelu) for i in x]
ax.plot(x, y_gelu, label="GELU function", linewidth=2, color='lightgray')
for i in range(len(ranges_gelu)):
    x_piecewise = np.linspace(ranges_gelu[i][0], ranges_gelu[i][1], 500)
    y_piecewise = [piecewise_linear_approx(j, [ranges_gelu[i]], [a_gelu[i]], [b_gelu[i]]) for j in x_piecewise]
    ax.plot(x_piecewise, y_piecewise, label=f"{a_gelu[i]}x + {b_gelu[i]} for {ranges_gelu[i]}", linestyle='dashed', color=colors[i])
    ax.plot(ranges_gelu[i][0], piecewise_linear_approx(ranges_gelu[i][0], [ranges_gelu[i]], [a_gelu[i]], [b_gelu[i]]), 'ko', markersize=5)  # lower bound
    ax.plot(ranges_gelu[i][1], piecewise_linear_approx(ranges_gelu[i][1], [ranges_gelu[i]], [a_gelu[i]], [b_gelu[i]]), 'ko', markersize=5)  # upper bound
    # adjust_text_position(ranges_gelu[i][0], piecewise_linear_approx(ranges_gelu[i][0], [ranges_gelu[i]], [a_gelu[i]], [b_gelu[i]]), f'({ranges_gelu[i][0]:.2f}, {piecewise_linear_approx(ranges_gelu[i][0], [ranges_gelu[i]], [a_gelu[i]], [b_gelu[i]]):.2f})', ax)  # lower bound coordinate
    # adjust_text_position(ranges_gelu[i][1], piecewise_linear_approx(ranges_gelu[i][1], [ranges_gelu[i]], [a_gelu[i]], [b_gelu[i]]), f'({ranges_gelu[i][1]:.2f}, {piecewise_linear_approx(ranges_gelu[i][1], [ranges_gelu[i]], [a_gelu[i]], [b_gelu[i]]):.2f})', ax)  # upper bound coordinate
ax.set_title("GELU Function and Its Piecewise Linear Approximations")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
ax.grid(True)
plt.tight_layout()
plt.show()


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
        # check if b_list[i] is negative, if so, change the sign to be - instead of +
        if b_list[i] < 0:
            ax.plot(x_piecewise, y_piecewise, label=f"{a_list[i]}x - {abs(b_list[i])} for {ranges[i]}", linestyle='dashed',
                    color=colors[i])
        else:
            ax.plot(x_piecewise, y_piecewise, label=f"{a_list[i]}x + {b_list[i]} for {ranges[i]}", linestyle='dashed',
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

def chart_function_and_approximations_legend_top_left(function, function_name, ranges, a_list, b_list, save_file_name=None):
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
        # check if b_list[i] is negative, if so, change the sign to be - instead of +
        if b_list[i] < 0:
            ax.plot(x_piecewise, y_piecewise, label=f"{a_list[i]}x - {abs(b_list[i])} for {ranges[i]}",
                    linestyle='dashed',
                    color=colors[i])
        else:
            ax.plot(x_piecewise, y_piecewise, label=f"{a_list[i]}x + {b_list[i]} for {ranges[i]}", linestyle='dashed',
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
    # make legend text size 1.5 times larger than default
    ax.legend(loc='upper left',fontsize='large')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    if save_file_name is not None:
        fig.savefig(save_file_name, bbox_inches='tight', dpi=300)


# Modify the plotting function to allocate more horizontal space for the left plot and make the right plot slimmer
def chart_function_and_approximations_with_custom_size(function, function_name, ranges, a_list, b_list, x_zoom, y_zoom,
                                                       left_width, right_width, save_file_name=None):
    range_minimum = ranges[0][0]
    range_maximum = ranges[-1][1]
    x = np.linspace(range_minimum, range_maximum, 1000)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(left_width + right_width, 6),
                                   gridspec_kw={'width_ratios': [left_width, right_width]})

    # First plot: Function and approximations
    y = function(x)
    y_approx = [piecewise_linear_approx(i, ranges, a_list, b_list) for i in x]

    ax1.plot(x, y, label=f"{function_name} function", linewidth=2, color='darkgray')
    for i in range(len(ranges)):
        x_piecewise = np.linspace(ranges[i][0], ranges[i][1], 500)
        y_piecewise = [piecewise_linear_approx(j, [ranges[i]], [a_list[i]], [b_list[i]]) for j in x_piecewise]
        if b_list[i] < 0:
            ax1.plot(x_piecewise, y_piecewise, label=f"{a_list[i]}x - {abs(b_list[i])} for {ranges[i]}",
                     linestyle='dashed',
                     color=colors[i])
        else:
            ax1.plot(x_piecewise, y_piecewise, label=f"{a_list[i]}x + {b_list[i]} for {ranges[i]}", linestyle='dashed',
                     color=colors[i])
        ax1.plot(ranges[i][0], piecewise_linear_approx(ranges[i][0], [ranges[i]], [a_list[i]], [b_list[i]]), 'ko',
                 markersize=4)  # lower bound
        ax1.plot(ranges[i][1], piecewise_linear_approx(ranges[i][1], [ranges[i]], [a_list[i]], [b_list[i]]), 'ko',
                 markersize=4)  # upper bound
    ax1.set_title(f"{function_name} Function and Its Piecewise Linear Approximations")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc='upper left', fontsize='large')
    ax1.grid(True)

    # Create an inset axis to show the zoomed-in region in the first plot
    axins = ax1.inset_axes([0.45, 0.45, 0.52, 0.42])
    axins.plot(x, y, linewidth=2, color='darkgray')
    for i in range(len(ranges)):
        x_piecewise = np.linspace(ranges[i][0], ranges[i][1], 500)
        y_piecewise = [piecewise_linear_approx(j, [ranges[i]], [a_list[i]], [b_list[i]]) for j in x_piecewise]
        axins.plot(x_piecewise, y_piecewise, linestyle='dashed', color=colors[i], linewidth=1)
    axins.set_xlim(x_zoom)
    axins.set_ylim(y_zoom)
    ax1.indicate_inset_zoom(axins)

    # Second plot: Absolute Error
    abs_error = np.abs(y - np.array(y_approx))
    ax2.plot(x, abs_error, label="Absolute Error", color='r')
    ax2.set_title(f"Absolute Error of {function_name} Approximation")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Absolute Error")
    ax2.grid(True)

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

# With different legend locations
# Sigmoid
chart_function_and_approximations_legend_top_left(sigmoid, "Sigmoid", ranges_sigmoid, a_sigmoid, b_sigmoid, "sigmoid_approximation_visualized_integrated_legend.png")
# SiLU
chart_function_and_approximations_legend_top_left(silu, "SiLU", ranges_silu, a_silu, b_silu, "silu_approximation_visualized_integrated_legend.png")
# GELU
chart_function_and_approximations_legend_top_left(gelu, "GELU", ranges_gelu, a_gelu, b_gelu, "gelu_approximation_visualized_integrated_legend.png")

# Plot the GELU function with custom plot sizes
chart_function_and_approximations_with_custom_size(gelu, "GELU", ranges_gelu, a_gelu, b_gelu, x_zoom=(-2.5, 0.2),
                                                   y_zoom=(-0.26, 0.116), left_width=12, right_width=6,
                                                   save_file_name='gelu_approximation_visualized_zoom_custom_size.png')