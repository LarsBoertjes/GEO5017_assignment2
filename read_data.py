import numpy as np
import os

def get_pointclouds():
    def get_label(index):
        if 0 <= index < 100:
            return 'building'
        elif 100 <= index < 200:
            return 'car'
        elif 200 <= index < 300:
            return 'fence'
        elif 300 <= index < 400:
            return 'pole'
        elif 400 <= index < 500:
            return 'tree'
        else:
            return None

    point_clouds = {
        'building': [],
        'car': [],
        'fence': [],
        'pole': [],
        'tree': []
    }

    for i in range(500):
        filename = f"data/{i:03d}.xyz"
        with open(filename, 'r') as xyz:
            lines = xyz.readlines()
            pc = np.array([line.split() for line in lines], dtype=float)
            label = get_label(i)
            point_clouds[label].append({'point_cloud': pc})

    return point_clouds


def read_point_clouds(folder_path):
    point_clouds = []
    labels = []

    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r') as file:
            points = []
            lines = file.readlines()
            for line in lines:
                coords = line.strip().split()
                x = float(coords[0])
                y = float(coords[1])
                z = float(coords[2])

                point = np.array([x, y, z])  # Pass coordinates as a list
                points.append(point)

            point_clouds.append(points)

            label = int(filename[:3])

            if 0 <= label <= 99:
                labels.append("building")
            elif 100 <= label <= 199:
                labels.append("car")
            elif 200 <= label <= 299:
                labels.append("fence")
            elif 300 <= label <= 399:
                labels.append("pole")
            elif 400 <= label <= 499:
                labels.append("tree")

    return point_clouds, labels,