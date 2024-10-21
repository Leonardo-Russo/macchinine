import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random

from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.rotations import active_matrix_from_intrinsic_euler_xyz, matrix_from_axis_angle
from pytransform3d.transformations import transform_from, plot_transform
from pytransform3d.camera import make_world_grid, world2image, plot_camera

def is_approx_rotation_matrix(R, tolerance=1e-4):
    if R.shape != (3, 3):
        raise ValueError(f"must be a 3x3 matrix:, instead  shape {R.shape=} {R=} ")
    
    # Check if the matrix is close to orthogonal: R.T * R ≈ I
    I_approx = np.dot(R.T, R)
    identity_matrix = np.eye(3)
    if not np.allclose(I_approx, identity_matrix, atol=tolerance):
        return False

    # Check if the determinant of the matrix is close to 1
    if not np.isclose(np.linalg.det(R), 1, atol=tolerance):
        return False

    return True

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

def generate_random_camera_position(xlim=(0, 0), ylim=(0, 0), zlim=(5, 5)):
    """ Generate random box with the given limit. """
    # Generate Camera Position
    x = random.uniform(xlim[0], xlim[1])
    y = random.uniform(ylim[0], ylim[1])
    z = random.uniform(zlim[0], zlim[1])
    camera_position = np.array([x, y, z])

    # Retrieve alpha and beta angles
    R, _ = lookat(camera_position, np.array([0, 0, 0]), np.array([0, 0, 1]))
    R = R @ matrix_from_axis_angle((1, 0, 0, np.pi))
    r_hat = R[:, 0]     # x-axis of the camera frame
    # r_hat = np.array([R[0][2], R[1][2], R[2][2]])
    alpha = np.arctan2(r_hat[1], r_hat[0])          # TODO: this is wrong but for now I don't care cause I don't use them !!!
    beta = np.arcsin(r_hat[2])

    return camera_position, alpha, beta


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
        center = np.array([[
            np.nan,
            np.nan
        ]])

        return [top_left, top_right, bottom_right, bottom_left], center
    
    else:
        center=np.array([[
            int((x_max+x_min)/2),
            int((y_max+y_min)/2)
        ]])

    return [top_left, top_right, bottom_right, bottom_left], center


def getAzimuthElevation(focal_length, sensor_size, image_size, point_on_image, R_C2W, check_flag= True):
    """
    Retrieve Azimuth and Elevation angles from the bounding box center coordinates on image plane.
    """
    # Calculate K Matrix
    fx = (focal_length / sensor_size[0]) * image_size[0]
    fy = (focal_length / sensor_size[1]) * image_size[1]
    cx = image_size[0] / 2
    cy = image_size[1] / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    # Compute r_hat
    r = np.array([point_on_image[0][0], point_on_image[0][1], 1])       # expanded image plane coordinates [u, v, 1]
    r_hat_C = normalize(np.linalg.inv(K) @ r)                           # r_hat in camera reference frame
    r_hat = R_C2W @ r_hat_C                                             # r_hat in world reference frame

    # Calculate Azimuth and Elevation
    azimuth = np.arctan2(r_hat[1], r_hat[0])
    elevation = np.arcsin(r_hat[2])

    return azimuth, elevation, r_hat

def findRoadIntersection(camera_position, direction):
    """
    Find the intersection of a line and the z=0 plane in world coordinates.

    Parameters:
    camera_position (numpy.ndarray): The position of the camera in world coordinates.
    b_hat (numpy.ndarray): The unit direction vector from the camera.

    Returns:
    intersection (numpy.ndarray): The point of intersection on the z=0 plane.
    """    

    camera_position=camera_position.reshape(3)
    c_z = camera_position[2]
    b_z = direction[2]

    # Calculate the scalar parameter for the intersection
    t = -c_z / b_z

    # Calculate the intersection point
    intersection = camera_position[:2] + t * direction[:2]

    return intersection

def getImage(debug=False, focal_length=0.0036, sensor_size=(0.00367, 0.00274), image_size=(640, 480), kappa=0.4):
    """
    Generate an image with specified camera settings and visualize the resulting scene.

    Parameters:
    debug (bool): Enables debugging mode to visualize additional elements.
    focal_length (float): The focal length of the camera in meters.
    sensor_size (tuple): The size of the camera sensor in meters (width, height).
    image_size (tuple): The size of the image in pixels (width, height).

    Returns:
    dict: A dictionary containing information about the camera setup and the generated image.
    """

    # Define Camera Position and Orientation
    camera_position, alpha, beta = generate_random_camera_position(xlim=(-5, 5), ylim=(-5, 5), zlim=(5, 5))

    # Define vertical direction
    up = np.array([0, 0, 1])

    # Define 'from' and 'to' points for the camera
    from_point = camera_position
    to_point = np.array([0, 0, 0])    # the camera always points at the origin of the world coordinates

    scene_dim = 25                  # this represents the size of the square in which we intend to place the vehicles around the camera

    # Create the bounding box, be shure that bb_center is positive and inside the image plane
    good_sample = False
    max_samples = 1000
    samples_counter = 1
    while not good_sample:
        box, center = generate_random_box(hlim=(1.5, 2), wlim=(3, 4), llim=(5, 8), xlim=(-scene_dim, scene_dim), ylim=(-scene_dim, scene_dim), zlim=(0, 0))

        # Compute Rotation and Translation -> Transformation Matrix
        R_C2W, t_C2W = lookat(from_point, to_point, up)     # these are the rotation and translation matrices
        R_C2W = R_C2W @ matrix_from_axis_angle((1, 0, 0, np.pi))     # flips about axis 1 to obtain Camera Frame
        cam2world = transform_from(R_C2W, t_C2W)

        # Create intrinsic camera matrix
        intrinsic_camera_matrix = np.array([
            [focal_length, 0, sensor_size[0] / 2],
            [0, focal_length, sensor_size[1] / 2],
            [0, 0, 1]
        ])

        # Transform from World to Image
        image_box = world2image(box, cam2world, sensor_size, image_size, focal_length, kappa=kappa)
        image_center = world2image([center], cam2world, sensor_size, image_size, focal_length, kappa=kappa)
        
        # Compute the bounding box and its center
        bounding_box, bb_center = create_bounding_box(image_box)

        if  np.all(bb_center > 0) and np.all(bb_center < image_size) :
            good_sample = True

        else:
            samples_counter += 1

            if debug and good_sample:
                world_grid = make_world_grid(n_lines=11, n_points_per_line=101, xlim=[-10, 10], ylim=[-10, 10])
                image_grid = world2image(world_grid, cam2world, sensor_size, image_size, focal_length, kappa=kappa)
                zc = np.array([R_C2W[0][2], R_C2W[1][2], R_C2W[2][2]])    # zc versor -> pointing direction of the camera
                _, _, b_hat = getAzimuthElevation(focal_length, sensor_size, image_size, bb_center, R_C2W)
                visualize_scene(image_size, beta, alpha, box, world_grid, camera_position, image_grid, image_box, bounding_box, image_center, bb_center, zc, b_hat)

            if samples_counter > max_samples:
                raise ValueError("Cannot find a bounding box center that is inside the image plane.")
                break

    # Compute Bounding Box Center's Azimuth and Elevation
    azimuth, elevation, b_hat = getAzimuthElevation(focal_length, sensor_size, image_size, bb_center, R_C2W)
    phi = np.pi/2 - elevation

    if elevation > 0:
        debug = True            # this is a case that should never happen

    testing = 0
    if testing:
      test_world0 = [[5, 5, 0, 1]]
      test_image = world2image(test_world0, cam2world, sensor_size, image_size, focal_length, kappa=kappa)
      _, _, test_hat = getAzimuthElevation(focal_length, sensor_size, image_size, test_image, R_C2W)
      test_world1 = findRoadIntersection(camera_position, test_hat)
      print(test_world0[0][:3], [test_world1.tolist()[0], test_world1.tolist()[1], 0])


    if debug:
        # Visualize in debug mode
        world_grid = make_world_grid(n_lines=11, n_points_per_line=101, xlim=[-10, 10], ylim=[-10, 10])
        image_grid = world2image(world_grid, cam2world, sensor_size, image_size, focal_length, kappa=kappa)

        # Set up and show figure
        zc = np.array([R_C2W[0][2], R_C2W[1][2], R_C2W[2][2]])    # zc versor -> pointing direction of the camera
        visualize_scene(image_size, beta, alpha, box, world_grid, camera_position, image_grid, image_box, bounding_box, image_center, bb_center, zc, b_hat)

    # Group all the info for a sample inside a dictionary
    sample = {
        "camera_position": camera_position,
        "bounding_box": bounding_box,
        "bb_center": bb_center,
        "image_center": image_center,       # 2D coordinates of the bb center projected on the ground
        "image_size": image_size,
        "focal_length": focal_length,
        'sensor_size': sensor_size,
        'image_size' : image_size,
        'kappa': kappa,
        'r': R_C2W,
        "true_center": center[:2]           # this is the true CM of the vehicle projected onto the road plane a.k.a. what we want to predict
    }

    return sample

def visualize_scene(image_size, elevation, azimuth, box, world_grid, camera_position, image_grid, image_box, bounding_box, image_center, bb_center, zc, b_hat):
    """
    Visualize the scene with the generated elements.

    Parameters:
    image_size, beta, azimuth, box, world_grid, camera_position, image_grid, image_box, bounding_box, image_center, bb_center
    """

    plt.figure(figsize=(12, 5))

    # Show the World Image
    ax = make_3d_axis(1, 121, unit="m")
    plot_transform(ax)
    ax.scatter(box[:, 0], box[:, 1], box[:, 2], s=2, alpha=1, c='r')
    ax.scatter(world_grid[:, 0], world_grid[:, 1], world_grid[:, 2], s=1, alpha=0.2)
    ax.scatter(0, 0, 0, s=20, alpha=1, c='g')
    ax.quiver(camera_position[0], camera_position[1], camera_position[2], b_hat[0], b_hat[1], b_hat[2], length=8, color='red', normalize=True)    # camera pointing direction
    ax.quiver(camera_position[0], camera_position[1], camera_position[2], zc[0], zc[1], zc[2], length=5, color='blue', normalize=True)            # from camera to bb_center direction

    lim = (-np.max(np.abs(camera_position)) / 2, np.max(np.abs(camera_position)) / 2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_zlim(lim)
    ax.set_title("Camera and world frames")
    ax.view_init(elev=np.rad2deg(elevation), azim=np.rad2deg(azimuth))

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