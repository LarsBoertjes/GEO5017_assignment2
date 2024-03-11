import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spatial
from read_data import get_pointclouds
from mpl_toolkits.mplot3d import Axes3D
import compas
from sklearn.decomposition import PCA

point_clouds = get_pointclouds()

# print(pc)
# # minimum oriented bounding box
# # obb = compas.geometry.oriented_bounding_box_numpy(coords)
# # compas.



for label, clouds in point_clouds.items():
    # i = 0
    for cloud in clouds:
        pc = cloud['point_cloud']
        height = np.max(pc[:, 2])
        height_object = np.max(pc[:, 2]) - np.min(pc[:, 2])
        area_bbox = (np.max(pc[:, 1]) - np.min(pc[:, 1])) * (np.max(pc[:, 0]) - np.min(pc[:, 0]))
        hull = spatial.ConvexHull(pc[:, 0:2])
        num_pts = pc.shape[0]
        # plt.plot(pc[:, 0], pc[:, 1], 'o')
        # plt.plot(pc[hull.vertices, 0], pc[hull.vertices, 1], 'r--', lw=2)
        # plt.plot(pc[hull.vertices[0], 0], pc[hull.vertices[0], 1], 'ro')
        # plt.title(f"{i}")
        # plt.show()
        cloud['height'] = height
        cloud['height_object'] = height_object
        cloud['area'] = area_bbox
        cloud['convex_area'] = hull.volume
        cloud['num_pts'] = num_pts
        # i += 1
        # cloud['convex_volume'] = hull.volume


# for label, clouds in point_clouds.items():
#     heights = []
#     areas = []
#     for cloud in clouds:
#         heights.append(cloud['height_object'])
#         areas.append(cloud['convex_area'])
#     plt.scatter(heights, areas, label=label)
#
# plt.xlabel('Relative height')
# plt.ylabel('Area')
# plt.legend()
#
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for label, clouds in point_clouds.items():
    heights = []
    areas = []
    points = []
    for cloud in clouds:
        if cloud['num_pts'] > 20000:
            continue
        heights.append(cloud['height_object'])
        areas.append(cloud['convex_area'])
        points.append(cloud['num_pts'])

    ax.scatter(points, areas, heights, label=label)

ax.set_xlabel('number of points')
ax.set_ylabel('Convex Area')
ax.set_zlabel('relative height object')
plt.legend()
plt.show()