import numpy as np
import math
from read_data import read_point_clouds, get_pointclouds
from plotting_features import plot_distributions
import scipy.spatial as spatial
from bbox import minBoundingRect


def geometric_features(pc):
    # computing some basic geometric features
    height = np.max(pc[:, 2])
    height_object = np.max(pc[:, 2]) - np.min(pc[:, 2])
    num_pts = pc.shape[0]
    area_bbox = (np.max(pc[:, 1]) - np.min(pc[:, 1])) * (np.max(pc[:, 0]) - np.min(pc[:, 0]))

    # computing with the minimum oriented bounding box based on convex hull
    hull = spatial.ConvexHull(pc[:, 0:2])
    mbbox = minBoundingRect(hull.points)
    area_oriented = mbbox[1]

    sides = mbbox[2], mbbox[3]
    aspect_ratio = max(sides) / min(sides)

    avg_density_m2 = num_pts / mbbox[1]
    avg_density_m3 = num_pts / (mbbox[1] * height_object)

    # convex hull
    convex_area = hull.volume
    perimeter = hull.area
    shape_index = 1.0 * convex_area / perimeter

    # computing differences between oriented bounding box centre and point cloud centres
    distance_z = np.mean(pc[:, 2]) - ((np.max(pc[:, 2]) + np.min(pc[:, 2])) / 2)
    center_xy = mbbox[4]
    avg_xy = np.mean(pc[:, 0]), np.mean(pc[:, 1])
    distance_xy = abs(math.dist(center_xy, avg_xy))

    return (height_object, area_bbox, area_oriented, aspect_ratio, avg_density_m2, avg_density_m3, convex_area,
            shape_index, distance_z, distance_xy)

def extract_geometric_features():
    point_clouds = get_pointclouds()
    (height_object, area_bbox, area_oriented, aspect_ratio, avg_density_m2, avg_density_m3, convex_area,
     diff_con_mbbox, distance_z, distance_xy) = [], [], [], [], [], [], [], [], [], []

    features = []

    for label, clouds in point_clouds.items():
        for cloud in clouds:
            pc = cloud['point_cloud']
            feat = geometric_features(pc)
            height_object.append(feat[0])
            area_bbox.append(feat[1])
            area_oriented.append(feat[2])
            aspect_ratio.append(feat[3])
            avg_density_m2.append(feat[4])
            avg_density_m3.append(feat[5])
            convex_area.append(feat[6])
            diff_con_mbbox.append(feat[7])
            distance_z.append(feat[8])
            distance_xy.append(feat[9])

    features.append(height_object)
    features.append(area_bbox)
    features.append(area_oriented)
    features.append(aspect_ratio)
    features.append(avg_density_m2)
    features.append(avg_density_m3)
    features.append(convex_area)
    features.append(diff_con_mbbox)
    features.append(distance_z)
    features.append(distance_xy)

    return features

# # compute geometric features
# (height_object, area_bbox, area_oriented, aspect_ratio, avg_density_m2, avg_density_m3, convex_area,
#  diff_con_mbbox, distance_z, distance_xy) = [], [], [], [], [], [], [], [], [], []
#
# labels = []
#
# for label, clouds in point_clouds.items():
#     for cloud in clouds:
#         pc = cloud['point_cloud']
#         features = geometric_features_not_eigen(pc)
#         height_object.append(features[0])
#         area_bbox.append(features[1])
#         area_oriented.append(features[2])
#         aspect_ratio.append(features[3])
#         avg_density_m2.append(features[4])
#         avg_density_m3.append(features[5])
#         convex_area.append(features[6])
#         diff_con_mbbox.append(features[7])
#         distance_z.append(features[8])
#         distance_xy.append(features[9])
#         labels.append(label)
#
#
# # plot the feature distributions for each label
# features = [height_object, area_bbox, area_oriented, aspect_ratio, avg_density_m2, avg_density_m3, convex_area,
#  diff_con_mbbox, distance_z, distance_xy]
# feature_names = ['Relative Height', 'Area Bounding Box', 'Area Oriented Bounding Box', 'Aspect Ratio',
#                  'Density m^2', 'Density m^3', 'Area Convex Hull', 'Ratio Convex Hull - Oriented Bounding Box',
#                  'Distance Z', 'Distance XY']
# plot_distributions(features, labels, feature_names)
