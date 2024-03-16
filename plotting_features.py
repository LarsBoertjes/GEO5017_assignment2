import numpy as np
import matplotlib.pyplot as plt


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