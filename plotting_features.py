import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distributions(features, labels, feature_names):
    ranges = [(0,100), (100, 200), (200, 300), (300, 400), (400, 500)]

    fig, axs = plt.subplots(5, len(features), figsize=(10*len(features), 20), sharex=False)

    normalized_values = np.linspace(0, 1, len(features))

    viridis_cmap = plt.get_cmap('viridis')

    colors = viridis_cmap(normalized_values)

    for j, (feature, feature_name, color) in enumerate(zip(features, feature_names, colors)):
        overall_min = min(min(feature[start:end]) for start, end in ranges)
        overall_max = max(max(feature[start:end]) for start, end in ranges)
        bin_edges = np.linspace(overall_min, overall_max, num=21)

        # Compute the maximum frequency for the current feature
        max_frequency = 0
        for i, (start, end) in enumerate(ranges):
            subset_frequency, _ = np.histogram(feature[start:end], bins=bin_edges)
            max_frequency = max(max_frequency, max(subset_frequency))

        for i, (start, end) in enumerate(ranges):
            frequencies, _ = np.histogram(feature[start:end], bins=bin_edges)
            axs[i, j].hist(feature[start:end], bins=bin_edges, color=color, edgecolor='black')
            label = labels[start]
            axs[i, j].set_title(f'{feature_name} for {label}')
            axs[i, j].set_xlabel(feature_name)
            axs[i, j].set_ylabel('Frequency')
            axs[i, j].set_ylim(0, max_frequency + max_frequency * 0.1)  # Set y-axis limit for the current subplot

    plt.tight_layout()
    plt.show()


def plot_scatter_matrix(features, matrix):
    fig, ax = plt.subplots(figsize=(10, 8))

    cax = ax.matshow(matrix, cmap='viridis')
    fig.colorbar(cax)

    # set feature names as tick labels
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(features)
    ax.set_yticklabels(features)

    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    # Add text annotations for each cell in the heatmap with color contrast
    for (i, j), val in np.ndenumerate(matrix):
        color = 'white' if matrix[i, j] < (matrix.max() - matrix.min()) / 2 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

    # Set the title and display the plot
    plt.title('Scatter Matrix')
    plt.show()


def plot_scatter_matrices(features, within_class_matrix, between_class_matrix):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    # Plot Within-Class Scatter Matrix
    cax1 = axes[0].matshow(within_class_matrix, cmap='viridis')
    fig.colorbar(cax1, ax=axes[0])
    axes[0].set_title('Within-Class Scatter Matrix')

    # Plot Between-Class Scatter Matrix
    cax2 = axes[1].matshow(between_class_matrix, cmap='viridis')
    fig.colorbar(cax2, ax=axes[1])
    axes[1].set_title('Between-Class Scatter Matrix')

    # Setting up the feature names as tick labels for both plots
    for ax in axes:
        ax.set_xticks(np.arange(len(features)))
        ax.set_yticks(np.arange(len(features)))
        ax.set_xticklabels(features)
        ax.set_yticklabels(features)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations with color contrast
    for matrix, ax in zip([within_class_matrix, between_class_matrix], axes):
        for (i, j), val in np.ndenumerate(matrix):
            color = 'white' if val < (matrix.max() - matrix.min()) / 2 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

    plt.tight_layout()
    plt.show()