import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distributions(features, labels, feature_names):
    ranges = [(0,100), (100, 200), (200, 300), (300, 400), (400, 500)]
    classes = ['building', 'car', 'fence', 'pole', 'tree']

    num_features = features.shape[1]
    fig, axs = plt.subplots(5, num_features, figsize=(10*num_features, 20), sharex=False)

    normalized_values = np.linspace(0, 1, num_features)
    viridis_cmap = plt.get_cmap('viridis')
    colors = viridis_cmap(normalized_values)

    for j in range(num_features):
        feature = features[:, j]
        feature_name = feature_names[j]
        color = colors[j]

        # Determine overall min and max for the feature for bin edges
        overall_min = np.min(feature)
        overall_max = np.max(feature)
        bin_edges = np.linspace(overall_min, overall_max, num=21)

        max_frequency = 0
        for start, end in ranges:
            # Find the max frequency to set y-axis limits consistently
            subset_frequency, _ = np.histogram(feature[start:end], bins=bin_edges)
            max_frequency = max(max_frequency, max(subset_frequency))

        for i, (start, end) in enumerate(ranges):
            frequencies, _ = np.histogram(feature[start:end], bins=bin_edges)
            axs[i, j].hist(feature[start:end], bins=bin_edges, color=color, edgecolor='black')
            unique_labels = np.unique(labels[start:end])
            if len(unique_labels) == 1:
                label_name = classes[unique_labels[0]]  # Convert label to class name
            else:
                label_name = 'Mixed'
            axs[i, j].set_title(f'{feature_name} for label {label_name}')
            axs[i, j].set_xlabel(feature_name)
            axs[i, j].set_ylabel('Frequency')
            axs[i, j].set_ylim(0, max_frequency + max_frequency * 0.1)

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
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=5)

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
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=5)

    plt.tight_layout()
    plt.show()


def plot_normalized_confusion_matrix(confusion_matrix, classifier_name):
    classes = ['building', 'car', 'fence', 'pole', 'tree']

    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', ax=ax, square=True,
                xticklabels=classes, yticklabels=classes)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'{classifier_name} Normalized Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()


def plot_overlap_matrix(overlap_matrix, class_labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(overlap_matrix, annot=True, xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
    plt.title('Feature Overlap Sum Matrix Between Classes')
    plt.show()

