import csv
import numpy as np
import random
from pytransform3d.transformations import transform_from
from pytransform3d.rotations import matrix_from_axis_angle

import utils

def load_csv_data(filepath):
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
    return data

def get_cam2world(camera_position):
    # Define vertical direction
    up = np.array([0, 0, 1])

    # Define 'from' and 'to' points for the camera
    from_point = camera_position
    to_point = np.array([0, 0, 0])    # the camera always points at the origin of the world coordinates

    # Compute Rotation and Translation -> Transformation Matrix
    R_C2W, t_C2W = utils.lookat(from_point, to_point, up)     # these are the rotation and translation matrices
    R_C2W = R_C2W @ matrix_from_axis_angle((1, 0, 0, np.pi))     # flips about axis 1 to obtain Camera Frame
    cam2world = transform_from(R_C2W, t_C2W)

    return cam2world

def generate_tracking_gt(data, camera_params):
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

        # DEBUG print the 2D bounding box of the vehicle
        print(f"2D Bounding Box of {obj_name}: {bbox_corners}")



if __name__ == "__main__":
    csv_filepath = r"C:\\Users\\Matteo2\\Documents\\Projects\\Macchinine\\SindToBlender\\LeoCode\\Blender-Rendering\eight_vehicles.csv"
    sind_data = load_csv_data(csv_filepath)

    # -- Get the camera parameters --
    # Calculate random view and camera position
    alpha, beta = utils.random_view()  # alpha, beta are respectively azimuth and elevation for the camera orientation
    camera_distance = random.uniform(20, 40)
    focal_length=0.0036
    sensor_size=(0.00367, 0.00274)
    image_size=(640, 480)
    kappa=0.4 # distortion parameter

    camera_position = camera_distance * np.array([
        np.cos(beta) * np.cos(alpha),
        np.cos(beta) * np.sin(alpha),
        np.sin(beta)
    ])

    # Get the camera-to-world transformation matrix
    cam2world = get_cam2world(camera_position)

    camera_params = {
        'cam2world': cam2world,
        'sensor_size': sensor_size,
        'image_size': image_size,
        'focal_length': focal_length,
        'kappa': kappa,
    }

    # -- Generate the ground truth tracking data --
    generate_tracking_gt(sind_data, camera_params)