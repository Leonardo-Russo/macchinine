import csv
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from pytransform3d.transformations import transform_from
from pytransform3d.rotations import matrix_from_axis_angle

import utils

def load_csv_data(filepath):
    """
    Load data from a CSV file and store it in a list of dictionaries.

    Each dictionary in the list represents a row in the CSV file, with the keys being the column names
    and the values being the data in each cell.

    Parameters
    ----------
    filepath : str
        The path to the CSV file.

    Returns
    -------
    list of dict
        The data from the CSV file, stored as a list of dictionaries.
    """

    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
    return data

def get_cam2world(camera_position, from_point, to_point):
    """
    Compute the transformation matrix from camera coordinates to world coordinates.

    The function first defines the vertical direction and the 'from' and 'to' points for the camera.
    It then computes the rotation and translation matrices using the lookat utility function.
    The rotation matrix is then flipped about axis 1 to obtain the Camera Frame.
    Finally, the transformation matrix is computed from the rotation and translation matrices.

    Parameters
    ----------
    camera_position : numpy.ndarray
        The position of the camera in 3D space.
    from_point : numpy.ndarray
        The starting point of the camera in 3D space.
    to_point : numpy.ndarray
        The target point that the camera is pointing at in 3D space.

    Returns
    -------
    numpy.ndarray
        The transformation matrix that converts camera coordinates to world coordinates.
    """
    # Define vertical direction
    up = np.array([0, 0, 1])

    # Compute Rotation and Translation -> Transformation Matrix
    R_C2W, t_C2W = utils.lookat(from_point, to_point, up)     # these are the rotation and translation matrices
    R_C2W = R_C2W @ matrix_from_axis_angle((1, 0, 0, np.pi))     # flips about axis 1 to obtain Camera Frame
    cam2world = transform_from(R_C2W, t_C2W)

    return cam2world

def generate_tracking_gt(data, camera_params, output_csv_path):
    """
    Generate ground truth tracking data from the given original SinD data and camera parameters.

    For each row in the data, the function loads information about the position, orientation, and size of the object.
    It then defines the 8 points of the box around the object and transforms these points from 3D world coordinates to 2D sensor coordinates.
    Finally, it obtains the 2D bounding box of the object.

    Parameters
    ----------
    data : list of dict
        The original SinD data for each object in each frame, stored as a list of dictionaries.

    camera_params : dict
        The parameters of the camera. This should be a dictionary with the following keys:
        - 'cam2world': The transformation matrix that converts camera coordinates to world coordinates.
        - 'sensor_size': The size of the camera sensor, given as a tuple of two floats (width, height).
        - 'image_size': The size of the image produced by the camera, given as a tuple of two integers (width, height).
        - 'focal_length': The focal length of the camera lens.
        - 'kappa': The distortion parameter of the camera.

    output_csv_path : str
        The path to save the generated tracking data as a CSV file.

    Returns
    -------
    None

    """

    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = list(data[0].keys()) + ['bbox_x_center', 'bbox_y_center', 'bbox_width', 'bbox_height']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            # Load info about position, orientation and size
            track_id = row['track_id']
            agent_type = row['agent_type']
            h = 1.8  # Assuming a height of 1.8 for simplicity
            x, y, z = (float(row['x']), float(row['y']), 0) # Assuming Z=0 for ground-level objects
            l, w = (float(row['length']), float(row['width']))
            yaw = float(row['yaw_rad'])
            
            obj_name = f"{agent_type}_{track_id}"

            # Define the 8 points of the box around the vehicle
            # TODO Add yaw information in the box points 3D coordinates 
            box_points = np.array([[x+l/2, y-w/2, 0, 1],
                        [x+l/2, y+w/2, 0, 1],
                        [x+l/2, y+w/2, z+h, 1],
                        [x+l/2, y-w/2, z+h, 1],
                        [x-l/2, y-w/2, 0, 1],
                        [x-l/2, y+w/2, 0, 1],
                        [x-l/2, y+w/2, z+h, 1],
                        [x-l/2, y-w/2, z+h, 1]])
            
            # Transform the box points and box center coords from 3D world coords to 2D sensor coords 
            image_box = utils.world2image(box_points, **camera_params)
            image_center = utils.world2image(np.array([[x, y, z, 1]]), **camera_params)

            # Obtain the 2D bounding box of the vehicle
            bbox_corners, bbox_center = utils.create_bounding_box(image_box)

            # Save the tracking data to the output CSV file, which contains
            # all the columns in the original SinD csv file, plus the 2D bounding box info,
            # which follows the YOLO format (x_center, y_center, width, height)
            bbox = {
                'bbox_x_center': bbox_center[0][0], 
                'bbox_y_center': bbox_center[0][1],
                'bbox_width': abs(bbox_corners[2][0] - bbox_corners[0][0]),
                'bbox_height': abs(bbox_corners[2][1] - bbox_corners[0][1])
            }

            ######## DEBUG #######
            plt.figure()
            ax = plt.gca()
            poly_image_box = [(x, image_size[1] - y) for x, y in utils.box2list(image_box)]
            ibox_patch = Polygon(poly_image_box, closed=True, edgecolor='#d6d327', fill=False, linewidth=1.5)
            ax.add_patch(ibox_patch)

            poly_bounding_box = [(x, image_size[1] - y) for x, y in bbox_corners]
            bbox_patch = Polygon(poly_bounding_box, closed=True, edgecolor='cyan', fill=False, linewidth=1.5)
            ax.add_patch(bbox_patch)
            ax.set_xlim(0, image_size[0])
            ax.set_ylim(0, image_size[1])
            plt.show(block=False)
            plt.pause(0.042)
            plt.close()
            #######################

            # Write the tracking data to the output CSV file
            writer.writerow({**row, 
                            **bbox})
            

if __name__ == "__main__":
    csv_filepath = "SinD/Data/8_02_1/first_vehicle_track.csv"
    sind_data = load_csv_data(csv_filepath)

    # -- Get the camera parameters --
    # Calculate random view and camera position
    alpha, beta = utils.random_view()  # alpha, beta are respectively azimuth and elevation for the camera orientation
    camera_distance  = random.uniform(20, 40)
    focal_length=0.0036
    sensor_size=(0.00367, 0.00274)
    image_size=(640, 480)
    kappa=0.4 # distortion parameter

    camera_position = camera_distance * np.array([
        np.cos(beta) * np.cos(alpha),
        np.cos(beta) * np.sin(alpha),
        np.sin(beta)
    ])

    # Define 'from' and 'to' points for the camera
    from_point = camera_position
    to_point = np.array([10, 10, 0])
    # Get the camera-to-world transformation matrix
    cam2world = get_cam2world(camera_position, from_point=from_point, to_point=to_point)

    camera_params = {
        'cam2world': cam2world,
        'sensor_size': sensor_size,
        'image_size': image_size,
        'focal_length': focal_length,
        'kappa': kappa,
    }

    # -- Generate the ground truth tracking data --
    output_csv_path = "tracking_gt.csv"
    generate_tracking_gt(sind_data, camera_params, output_csv_path=output_csv_path)