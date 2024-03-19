import numpy as np

def dataset():
    from read_data import point_clouds_for_classification
    from geometric_features import extract_geometric_features

    pc = point_clouds_for_classification('data')

    dataset = np.zeros((len(pc[1]), 4))

    all_features = extract_geometric_features()[0]

    relative_heights = []
    for height in pc[2]:
        relative_heights.append(max(height) - min(height))

    dataset[:, 0] = all_features[2]
    dataset[:, 1] = all_features[5]
    dataset[:, 2] = all_features[6]
    dataset[:, 3] = relative_heights

    labels = np.array(pc[1]).reshape(-1, 1)

    # Combine the dataset and labels for comprehensive storage
    combined_data = np.hstack((dataset, labels))

    # Write to a text file
    np.savetxt('dataset_labels.txt', combined_data, fmt='%f', header='Feature1 Feature2 Feature3 Feature4 Label', comments='')

    return dataset, labels

dataset()

def read_text_data(filename):
    # Load the data from the text file
    data = np.loadtxt(filename, skiprows=1)  # skiprows is used to skip the header

    # Separate the dataset into features and labels
    X = data[:, :-1]  # all columns except the last one
    y = data[:, -1].astype(int)  # last column, converted to integer

    return X, y

X, y = read_text_data('dataset_labels.txt')

print(X[:5])
print(y[:5])

print(len(X))
print(len(y))

print(len(X[0]))
