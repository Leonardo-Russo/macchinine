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

    # Check for NaN
    if np.isnan(x_min) or np.isnan(x_max) or np.isnan(y_min) or np.isnan(y_max):
        # print("Something in Bounding Box is NaN!"
        #       "x_min: ", x_min,
        #       "x_max: ", x_max,
        #       "y_min: ", y_min,
        #       "y_max: ", y_max)
        print("Points are: ", points)
        return None

    center=np.array([[
        int((x_max+x_min)/2),
        int((y_max+y_min)/2)
    ]])

    return [top_left, top_right, bottom_right, bottom_left], center


def random_view(alim=(0, 2*np.pi), blim=(np.deg2rad(10), np.deg2rad(60))):

    azimuth = random.uniform(alim[0], alim[1])
    beta = random.uniform(blim[0], blim[1])

    return azimuth, beta


def get_cam2world(from_point, to_point, up=np.array([0, 0, 1])):
    """
    Compute the transformation matrix from camera coordinates to world coordinates.

    The function first defines the vertical direction and the 'from' and 'to' points for the camera.
    It then computes the rotation and translation matrices using the lookat utility function.
    The rotation matrix is then flipped about axis 1 to obtain the Camera Frame.
    Finally, the transformation matrix is computed from the rotation and translation matrices.

    Parameters
    ----------
    from_point : numpy.ndarray
        The starting point of the camera in 3D space.
    to_point : numpy.ndarray
        The target point that the camera is pointing at in 3D space.
    up : numpy.ndarray
        Vector defining the vertical direction of the camera.

    Returns
    -------
    numpy.ndarray
        The transformation matrix that converts camera coordinates to world coordinates.
    """

    # Compute Rotation and Translation -> Transformation Matrix
    R_C2W, t_C2W = lookat(from_point, to_point, up)     # these are the rotation and translation matrices
    R_C2W = R_C2W @ matrix_from_axis_angle((1, 0, 0, np.pi))     # flips about axis 1 to obtain Camera Frame
    cam2world = transform_from(R_C2W, t_C2W)

    return cam2world

def visualize_scene(image_size, box, world_grid, camera_position, image_grid, image_box, bounding_box, image_center, bb_center):
    """
    Visualize the scene with the generated elements.

    Parameters:
    image_size, beta, azimuth, box, world_grid, camera_position, image_grid, image_box, bounding_box, image_center, bb_center
    """

    plt.figure(figsize=(12, 5))
    show_camera = False  # Set to True to show the camera in the plot

    # Show the World Image
    ax = make_3d_axis(1, 121, unit="m")
    plot_transform(ax)
    ax.scatter(box[:, 0], box[:, 1], box[:, 2], s=2, alpha=1, c='r')
    ax.scatter(world_grid[:, 0], world_grid[:, 1], world_grid[:, 2], s=1, alpha=0.2)
    ax.scatter(0, 0, 0, s=20, alpha=1, c='g')

    # if show_camera:
    #     plot_transform(ax, A2B=cam2world, s=0.3, name="Camera")
    #     plot_camera(ax, intrinsic_camera_matrix, cam2world, sensor_size=sensor_size, virtual_image_distance=0.5)

    lim = (-np.max(np.abs(camera_position)) / 2, np.max(np.abs(camera_position)) / 2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_zlim(lim)
    ax.set_title("Camera and world frames")
    # ax.view_init(elev=np.rad2deg(elevation), azim=np.rad2deg(azimuth))

    # Visualize Camera Image
    ax = plt.subplot(122, aspect="equal")
    ax.scatter(image_grid[:, 0], -(image_grid[:, 1] - image_size[1]), s=1, alpha=0.2)

    poly_image_box = [(x, image_size[1] - y) for x, y in box2list(image_box)]
    ibox = Polygon(poly_image_box, closed=True, edgecolor='#d6d327', fill=False, linewidth=1.5)
    ax.add_patch(ibox)

    poly_bounding_box = [(x, image_size[1] - y) for x, y in bounding_box]
    bbox = Polygon(poly_bounding_box, closed=True, edgecolor='cyan', fill=False, linewidth=1.5)
    ax.add_patch(bbox)
    ax.scatter(image_center[:, 0], -(image_center[:, 1] - image_size[1]))


    ax.scatter(bb_center[:, 0], -(bb_center[:, 1] - image_size[1]), edgecolor='cyan')

    ax.set_title("Camera image")
    ax.set_xlim(0, image_size[0])
    ax.set_ylim(0, image_size[1])

    plt.show()

def matrix2quat(R):
    """
    Convert a rotation matrix to a quaternion.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    q : ndarray, shape (4,)
        Quaternion (w, x, y, z).
    """
    assert R.shape == (3, 3), "Input rotation matrix must be 3x3"
    
    # Calculate the trace of the matrix
    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S = 4 * qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])


def C2q(C):
    """
    Convert a direction cosines matrix to quaternions.

    Parameters
    ----------
    C : array-like, shape (3, 3)
        Direction cosines matrix (rotation matrix).

    Returns
    -------
    q0 : float
        Scalar part of the quaternion.
    
    q : ndarray, shape (3,)
        Vector part of the quaternion.
    """
    assert C.shape == (3, 3), "Input matrix must be 3x3"

    q0 = 0.5 * np.sqrt(1 + C[0, 0] + C[1, 1] + C[2, 2])
    
    q = 1 / (4 * q0) * np.array([C[1, 2] - C[2, 1], 
                                 C[2, 0] - C[0, 2], 
                                 C[0, 1] - C[1, 0]])

    return np.array([q0, *q])