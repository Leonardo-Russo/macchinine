import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

import os
import matplotlib.pyplot as plt
from pytransform3d.rotations import plot_basis

from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.rotations import active_matrix_from_intrinsic_euler_xyz, matrix_from_axis_angle
from pytransform3d.transformations import transform_from, plot_transform
from pytransform3d.camera import make_world_grid, world2image, plot_camera


def generate_random_box(hlim=(2, 3), wlim=(3, 4), llim=(3, 5), xlim=(0, 0), ylim=(0, 0), zlim=(0, 0)):
    """ Generate random box with the given limit. """

    # Generate Box Dimensions
    h = random.uniform(hlim[0], hlim[1])
    w = random.uniform(wlim[0], wlim[1])
    l = random.uniform(llim[0], llim[1])

    # Generate Box Center
    x = random.uniform(xlim[0], xlim[1])
    y = random.uniform(ylim[0], ylim[1])
    z = random.uniform(zlim[0], zlim[1])

    points = np.array([[x+l/2, y-w/2, 0, 1],
                       [x+l/2, y+w/2, 0, 1],
                       [x+l/2, y+w/2, z+h, 1],
                       [x+l/2, y-w/2, z+h, 1],
                       [x-l/2, y-w/2, 0, 1],
                       [x-l/2, y+w/2, 0, 1],
                       [x-l/2, y+w/2, z+h, 1],
                       [x-l/2, y-w/2, z+h, 1]])

    center_on_road = [x, y, 0, 1]

    return points, center_on_road


def generate_box(h=2.5, w=3, l=4, x=0, y=0, z=0):
    """ Generate a box with the given parameters. """

    points = np.array([[x+l/2, y-w/2, 0, 1],
                       [x+l/2, y+w/2, 0, 1],
                       [x+l/2, y+w/2, z+h, 1],
                       [x+l/2, y-w/2, z+h, 1],
                       [x-l/2, y-w/2, 0, 1],
                       [x-l/2, y+w/2, 0, 1],
                       [x-l/2, y+w/2, z+h, 1],
                       [x-l/2, y-w/2, z+h, 1]])

    center_on_road=[x, y, 0, 1]

    return points, center_on_road


def box2list(box_corners):
    """" Utility function useful for plotting the box. """

    points = np.array([box_corners[0],
                       box_corners[1],
                       box_corners[2],
                       box_corners[3],
                       box_corners[0],
                       box_corners[4],
                       box_corners[5],
                       box_corners[6],
                       box_corners[7],
                       box_corners[4],
                       box_corners[5],
                       box_corners[1],
                       box_corners[2],
                       box_corners[6],
                       box_corners[7],
                       box_corners[3]])

    return points


def normalize(v):
    return v / np.linalg.norm(v)


def cross(a, b):
    return np.cross(a, b)


def lookat(from_point, to_point, up):
    forward = normalize(from_point - to_point)
    right = normalize(cross(up, forward))
    newup = cross(forward, right)

    r = np.array([right, newup, forward])
    t = np.array(from_point)
    return np.transpose(r), t


def create_bounding_box(points):
    # Convert string of points to numpy array
    points = np.array(points)

    # Find min and max values for x and y
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    # Constructing the four corners of the bounding box
    top_left = (x_min, y_min)
    top_right = (x_max, y_min)
    bottom_right = (x_max, y_max)
    bottom_left = (x_min, y_max)

    center=np.array([[
        int((x_max+x_min)/2),
        int((y_max+y_min)/2)
    ]])

    return [top_left, top_right, bottom_right, bottom_left], center


def random_view(alim=(0, 2*np.pi), blim=(np.deg2rad(10), np.deg2rad(60))):

    azimuth = random.uniform(alim[0], alim[1])
    beta = random.uniform(blim[0], blim[1])

    return azimuth, beta