import numpy as np
from os.path import exists
from eigen_features import extract_eigen_features
from geometric_features import extract_geometric_features


def feature_extraction(data_path):
    """
    Prepare features of the input point cloud objects
        data_path: the path to read data
    """
    # Check if the current data file exists
    data_file = data_path
    if not exists(data_file):
        print(f"{data_file} not found. Generating the file...")
        generate_data_file(data_file)
    else:
        return


def generate_data_file(data_file):
    data = extract_eigen_features()[0]
    data.extend(extract_geometric_features())
    labels = extract_eigen_features()[1]

    # Transform the output data
    outputs = np.transpose(np.array(data).astype(np.float32))

    # Create a list to store formatted data with labels
    formatted_data = []

    # Combine labels with corresponding rows of data and add IDs
    label_mapping = {'building': 0, 'car': 1, 'fence': 2, 'pole': 3, 'tree': 4}

    for idx, (label, row) in enumerate(zip(labels, outputs)):
        # Calculate ID based on the index with leading zeros
        ID = idx
        formatted_row = [ID, label_mapping[label]] + list(row)
        formatted_data.append(formatted_row)

    # Convert formatted_data to a NumPy array
    formatted_data_array = np.array(formatted_data)

    # convert formatted_data to a NumPy array
    formatted_data_array = np.array(formatted_data)

    # write the output to a local file
    data_header = ('ID, Label, Sum, Omnivariance, Eigenentropy, Linearity, Planarity, Sphericity, Anisotropy,'
                   'Relative Height, Area Bounding Box, Area Oriented Bounding Box, Aspect Ratio, Density m2,'
                   'Density m3, Area Convex Hull, Shape Index , Distance Z, Distance XY')
    np.savetxt(data_file, formatted_data_array, fmt='%10.5f', delimiter=',', newline='\n', header=data_header)


def data_loading(data_file='data.txt'):
    """
    Read the data with features from the data file
        data_file: the local file to read data with features and labels
    """
    # load data
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=',', comments='#')

    # extract object ID, feature X and label Y
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)

    return ID, X, y
