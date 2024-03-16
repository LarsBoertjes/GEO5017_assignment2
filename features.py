import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spatial
from read_data import get_pointclouds
from bbox import minBoundingRect
import math


point_clouds = get_pointclouds()


# print(pc)
# # minimum oriented bounding box
# # obb = compas.geometry.oriented_bounding_box_numpy(coords)
# # compas.


for label, clouds in point_clouds.items():
    for cloud in clouds:
        pc = cloud['point_cloud']
        height = np.max(pc[:, 2])
        height_object = np.max(pc[:, 2]) - np.min(pc[:, 2])
        distance_z = np.mean(pc[:, 2]) - (np.max(pc[:, 2]) + np.min(pc[:, 2]))
        area_bbox = (np.max(pc[:, 1]) - np.min(pc[:, 1])) * (np.max(pc[:, 0]) - np.min(pc[:, 0]))
        hull = spatial.ConvexHull(pc[:, 0:2])
        num_pts = pc.shape[0]
        mbbox = minBoundingRect(hull.points)
        sides = mbbox[2], mbbox[3]
        center_xy = mbbox[4]
        avg_xy = np.mean(pc[:, 0]), np.mean(pc[:, 1])
        distance_xy = abs(math.dist(center_xy, avg_xy))
        diff_con_mbbox = (hull.volume / area_bbox)

        avg_density_m2 = num_pts / mbbox[1]
        avg_density_m3 = num_pts / (mbbox[1] * height_object)
        aspect_ratio = max(sides) / min(sides)

        cloud['height'] = height
        cloud['height_object'] = height_object
        cloud['area'] = area_bbox
        cloud['convex_area'] = hull.volume
        cloud['density_m2'] = avg_density_m2
        cloud['density_m3'] = avg_density_m3
        cloud['aspect_ratio'] = aspect_ratio
        cloud['dis_xy'] = distance_xy
        cloud['dis_z'] = distance_z
        cloud['diff_area'] = diff_con_mbbox

        cloud['area_oriented'] = mbbox[1]

        cloud['mbbox'] = mbbox

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

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

for label, clouds in point_clouds.items():
    heights = []
    areas = []
    points = []
    for cloud in clouds:
        heights.append(cloud['height_object'])
        areas.append(cloud['dis_z'])
        points.append(cloud['density_m3'])

    ax.scatter(points, areas, heights, label=label)

# Set maximum values for each axis
# ax.set_xlim([0, 100])
# ax.set_ylim([0, 100])
# ax.set_zlim([0, 100])

ax.set_xlabel('density_m3')
ax.set_ylabel('area oriented')
ax.set_zlabel('relative height object')
plt.legend()
plt.show()