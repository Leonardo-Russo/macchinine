import bpy
import csv
import os
import json
from mathutils import Quaternion, Euler

def load_csv_data(filepath):
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            track_id = row['track_id']
            agent_type = row['agent_type']
            height = 1.8  # Default height for objects
            position = (float(row['x']), float(row['y']), height/2)  # Set z=0 for ground-level objects
            yaw = float(row['yaw_rad'])

            obj_name = f"{agent_type}_{track_id}"

            # Check if this track_id already has a corresponding Blender object
            if obj_name not in bpy.data.objects:
                bpy.ops.mesh.primitive_cube_add(location=position)
                obj = bpy.context.object
                obj.name = obj_name
                obj.scale = (float(row['length']), float(row['width']), height/2)
            else:
                obj = bpy.data.objects[obj_name]

            # Update object position and rotation
            obj.location = position
            obj.rotation_euler = Euler((0, 0, yaw), 'XYZ')

            # Keyframe insertion for animation
            frame_id = int(row['frame_id'])
            obj.keyframe_insert(data_path="location", frame=frame_id)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame_id)

def setup_camera(parameters_file="camera_parameters.json", additional_parameters_file="camera_additional_parameters.json"):
    parameters_path = os.path.join(os.getcwd(), parameters_file)
    additional_parameters_path = os.path.join(os.getcwd(), additional_parameters_file)

    with open(parameters_path, "r") as file:
        camera_parameters = json.load(file)

    with open(additional_parameters_path, "r") as file:
        additional_parameters = json.load(file)

    # Check if camera already exists, if not, create it
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.name = "Camera"
    else:
        camera = bpy.data.objects["Camera"]

    camera.data.type = 'PERSP'  # Set camera type to perspective

    # Set camera position
    camera.location = camera_parameters["camera_position"]

    # Set camera rotation (quaternion)
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = additional_parameters["camera_orientation"]

    # Set camera focal length
    camera.data.lens = camera_parameters["focal_length"]

    bpy.context.scene.camera = camera

def setup_lighting():
    # Delete default lights
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete(use_global=False)

    # light_positions = [
    #     (10, 10, 10),
    #     (-10, 10, 10),
    #     (-10, -10, 10),
    #     (10, -10, 10)
    # ]

    light_positions = [
        (-15, 0, 20),
    ] 

    for pos in light_positions:
        bpy.ops.object.light_add(type='SUN', location=pos)
        light = bpy.context.object
        light.data.energy = 10  # Adjust the energy of the lights

def apply_material_to_object(obj, color=(0.8, 0.8, 0.8, 1)):
    if not obj.data.materials:
        # Create a new material
        mat = bpy.data.materials.new(name=f"{obj.name}Material")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs['Base Color'].default_value = color  # Set color
        obj.data.materials.append(mat)

def set_background_color(color=(0.35, 0.35, 0.35, 1)):  # Dark gray background
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = color

def import_background(filepath, location=(20, 10, 0), scale=(0.07, 0.07, 0.07)):
    bpy.ops.wm.collada_import(filepath=filepath)
    background_obj = bpy.context.selected_objects[0]
    background_obj.name = "Background"
    background_obj.location = location
    background_obj.scale = scale
    bpy.ops.object.select_all(action='DESELECT')


# -- Load the prepared scene (background + camera + lighting) saved in a .blend file --
scene_filepath = os.path.join(os.getcwd(), "blender", "untitled.blend")
bpy.ops.wm.open_mainfile(filepath=scene_filepath)

# -- Load CSV Data -- #
#csv_filepath = r"G:\Il mio Drive\Codes\Python\Macchinine\SinD\Data\8_02_1\three_vehicles_track.csv"
csv_filepath = os.path.join(os.getcwd(), "SinD", "Data", "8_02_1", "three_vehicles_track.csv")
load_csv_data(csv_filepath)

# # -- Import Background -- #
# background_filepath = r"G:\Il mio Drive\Codes\Python\macchinine\blender\american-road-intersection\source\InterRoad.dae"
# import_background(background_filepath)

# # -- Setup Camera -- #
# setup_camera()

# # -- Setup Lighting -- #
# setup_lighting()

# -- Save the Scene -- #
output_filepath = os.path.join(os.getcwd(), "blender", "untitled.blend")
bpy.ops.wm.save_as_mainfile(filepath=output_filepath)
