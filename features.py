import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spatial
from read_data import get_pointclouds
import compas
from sklearn.decomposition import PCA

point_clouds = get_pointclouds()
print(point_clouds)

# print(pc)
# # minimum oriented bounding box
# # obb = compas.geometry.oriented_bounding_box_numpy(coords)
# # compas.

for label, clouds in point_clouds.items():
    for cloud in clouds:
        pc = cloud['point_cloud']
        height = np.max(pc[:, 2])
        height_object = np.max(pc[:, 2]) - np.min(pc[:, 2])
        area_bbox = (np.max(pc[:, 1]) - np.min(pc[:, 1])) * (np.max(pc[:, 0]) - np.min(pc[:, 0]))
        hull = spatial.ConvexHull(pc[:, -1])
        plt.plot(pc[:, 0], pc[:, 1], 'o')
        plt.plot(pc[hull.vertices, 0], pc[hull.vertices, 1], 'r--', lw=2)
        plt.plot(pc[hull.vertices[0], 0], pc[hull.vertices[0], 1], 'ro')
        plt.show()

        cloud['height'] = height
        cloud['height_object'] = height_object
        cloud['area'] = area_bbox
        cloud['convex_area'] = hull.area
        # cloud['convex_volume'] = hull.volume


print(point_clouds)
# PCA
# pca = PCA(n_components=3)
# pca.fit(pc)
#
# eigenvalues = pca.explained_variance_
#
# eigenvalues_sorted = np.sort(eigenvalues)[::-1]
#
# linearity = eigenvalues_sorted[0] / np.sum(eigenvalues_sorted)
#
# print("linearity:", linearity)

#plot
# for label, clouds in point_clouds.items():
#     heights = []
#     areas = []
#     for cloud in clouds:
#         heights.append(cloud['height_object'])
#         areas.append(cloud['convex_area'])
#     plt.scatter(heights, areas, label=label)
#
# # Set labels and legend
# plt.xlabel('Relative height')
# plt.ylabel('Area')
# plt.legend()
#
# # Show the plot
# plt.show()