import numpy as np

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