import bpy
import csv
from mathutils import Quaternion, Euler
import os

def load_csv_data_and_animate(filepath):
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            track_id = row['track_id']
            agent_type = row['agent_type']
            position = (float(row['x']), float(row['y']), 0)  # Assuming Z=0 for ground-level objects
            yaw = float(row['yaw_rad'])
            
            obj_name = f"{agent_type}_{track_id}"
            
            # Check if this track_id already has a corresponding Blender object
            if obj_name not in bpy.data.objects:
                bpy.ops.mesh.primitive_cube_add(location=position)
                obj = bpy.context.object
                obj.name = obj_name
                obj.scale = (float(row['length']), float(row['width']), 1.8)  # Assuming a height of 1.8 for simplicity
            else:
                obj = bpy.data.objects[obj_name]
            
            # Update object position and rotation
            obj.location = position
            obj.rotation_euler = Euler((0, 0, yaw), 'XYZ')
            
            # Keyframe insertion for animation
            frame_id = int(row['frame_id'])
            obj.keyframe_insert(data_path="location", frame=frame_id)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame_id)

    output_dir = os.getcwd()
    output_filepath = os.path.join(output_dir, "blender/untitled.blend")
    bpy.ops.wm.save_as_mainfile(filepath=output_filepath)

# Example usage
#csv_filepath = r"C:\\Users\\Matteo2\\Documents\\Projects\\Macchinine\\SindToBlender\\LeoCode\\Blender-Rendering\Veh_smoothed_tracks.csv"
#csv_filepath = r"C:\\Users\\Matteo2\\Documents\\Projects\\Macchinine\\SindToBlender\\LeoCode\\Blender-Rendering\Veh_smoothed_tracks_single.csv"
# csv_filepath = r"C:\\Users\\Matteo2\\Documents\\Projects\\Macchinine\\SindToBlender\\LeoCode\\Blender-Rendering\eight_vehicles.csv"
# csv_filepath = r"G:\Il mio Drive\Codes\Python\Macchinine\SinD\Data\8_02_1\first_vehicle_track.csv"
csv_filepath = r"G:\Il mio Drive\Codes\Python\Macchinine\SinD\Data\8_02_1\three_vehicles_track.csv"
load_csv_data_and_animate(csv_filepath)
