import csv
import numpy as np
import random
import cv2
import json
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from pytransform3d.transformations import transform_from
from pytransform3d.rotations import matrix_from_axis_angle
from pytransform3d.rotations import quaternion_from_matrix
import quaternion

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


def generate_groundtruth(data, camera_params, output_csv_path, debug=False):
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
        fieldnames = list(data[0].keys()) + ['bbox_x_center', 'bbox_y_center', 'bbox_width', 'bbox_height', 'x_center', 'y_center']
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
            box_points = np.array([[x+l/2, y-w/2, z, 1],
                        [x+l/2, y+w/2, z, 1],
                        [x+l/2, y+w/2, z+h, 1],
                        [x+l/2, y-w/2, z+h, 1],
                        [x-l/2, y-w/2, z, 1],
                        [x-l/2, y+w/2, z, 1],
                        [x-l/2, y+w/2, z+h, 1],
                        [x-l/2, y-w/2, z+h, 1]])
            
            # Transform the box points and box center coords from 3D world coords to 2D sensor coords 
            image_box = utils.world2image(box_points, cam2world=camera_params['cam2world'], sensor_size=camera_params['sensor_size'], image_size=camera_params['image_size'], focal_length=camera_params['focal_length'] * 1e-4, kappa=camera_params['kappa'])
            image_center = utils.world2image(np.array([[x, y, z, 1]]), cam2world=camera_params['cam2world'], sensor_size=camera_params['sensor_size'], image_size=camera_params['image_size'], focal_length=camera_params['focal_length'] * 1e-4, kappa=camera_params['kappa'])

            if np.isnan(image_box).any() or np.isnan(image_center).any():
                print(f"NaN values detected in image_box or image_center for object {obj_name} -> skipping this frame...")
                # this means that the object is not fully visible in the image and for this reason we cannot (for now) handle the 
                # computation of the center of the bounding box. For this reason, we'll skip this object and continue with the next one.
                
                bbox = {
                    'bbox_x_center': np.nan, 
                    'bbox_y_center': np.nan,
                    'bbox_width': np.nan,
                    'bbox_height': np.nan
                }

                center_on_road = {
                    'x_center': np.nan,
                    'y_center': np.nan
                }
                continue
            else:
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

                center_on_road = {
                    'x_center': image_center[0][0],
                    'y_center': image_center[0][1]
                }

                if debug:
                    # plt.figure()
                    # ax = plt.gca()
                    # poly_image_box = [(x, image_size[1] - y) for x, y in utils.box2list(image_box)]
                    # ibox_patch = Polygon(poly_image_box, closed=True, edgecolor='#d6d327', fill=False, linewidth=1.5)
                    # ax.add_patch(ibox_patch)

                    # poly_bounding_box = [(x, image_size[1] - y) for x, y in bbox_corners]
                    # bbox_patch = Polygon(poly_bounding_box, closed=True, edgecolor='cyan', fill=False, linewidth=1.5)
                    # ax.add_patch(bbox_patch)
                    # ax.set_xlim(0, image_size[0])
                    # ax.set_ylim(0, image_size[1])
                    # plt.show(block=False)
                    # plt.pause(0.092)
                    # plt.close()
                    
                    # Create a canvas to display the image and bounding box
                    canvas = 255*np.ones((image_size[1], image_size[0], 3), dtype=np.uint8)

                    # Draw the bounding box
                    cv2.polylines(canvas, [np.int32(bbox_corners)], isClosed=True, color=(255, 255, 0), thickness=2)

                    # Display the canvas
                    cv2.imshow("Tracking GT", canvas)
                    cv2.waitKey(92)  # Delay between frames in milliseconds\
            
            # Write the tracking data to the output CSV file
            writer.writerow({**row, 
                            **bbox,
                            **center_on_road})

        if debug:
        # Close the window after displaying the frame
            cv2.destroyAllWindows()
            

if __name__ == "__main__":
    csv_filepath = "SinD/Data/8_02_1/Veh_smoothed_tracks.csv"
    #csv_filepath = "SinD/Data/8_02_1/three_vehicles_track.csv"
    sind_data = load_csv_data(csv_filepath)

    # -- Get the camera parameters --
    with open('camera_parameters.json', 'r') as file:
        camera_params = json.load(file)

        camera_position = np.array(camera_params['camera_position'])     # position of the camera
        camera_target = np.array(camera_params['camera_target'])         # target point that the camera is pointing at
        focal_length = camera_params['focal_length'] * 1e-4              # focal length in m
        sensor_size = camera_params['sensor_size']                       # sensor size in mm
        image_size = camera_params['image_size']                         # image size in pixels
        kappa = camera_params['kappa']                                   # distortion parameter
    
    # Get the Camera-to-World Transformation matrix
    cam2world = utils.get_cam2world(from_point=camera_position, to_point=camera_target, up=np.array([0, 0, 1]))

    # Compute the quaternions from the rotation matrix - don't ask why this works but it doesn't work in other ways
    not_cam2world = utils.get_cam2world(from_point=camera_target, to_point=camera_position, up=np.array([0, 0, -1]))
    q = utils.C2q(not_cam2world[:3, :3].T)
    # q = utils.matrix2quat(not_cam2world[:3, :3])

    print(f"Camera Position: {camera_position}")
    print(f"Camera Target: {camera_target}")
    print(f"Focal Length: {focal_length}")
    print(f"Sensor Size: {sensor_size}")
    print(f"Image Size: {image_size}")
    print(f"Kappa: {kappa}")
    print(f"Camera-to-World Transformation Matrix:\n{cam2world}")
    print(f"Quaternions: {q}")
    
    # TODO: per qualche cazzo di motivo i quaternioni sono invertiti rispetto a quelli di blender

    camera_params['cam2world'] = cam2world

    camera_additional_params = {
        'cam2world': cam2world.tolist(),
        'camera_orientation': q.tolist()
    }

    # Save Additional Camera Parameters to a JSON file
    additional_camera_params_filepath = 'camera_additional_parameters.json'
    with open(additional_camera_params_filepath, 'w') as json_file:
        json.dump(camera_additional_params, json_file, indent=4)

    # -- Generate the ground truth tracking data --
    output_csv_path = "tracking_groundtruth.csv"
    generate_groundtruth(sind_data, camera_params, output_csv_path=output_csv_path, debug=False)
    print(f"Ground truth tracking data saved to {output_csv_path}")