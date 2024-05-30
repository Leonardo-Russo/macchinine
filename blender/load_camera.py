import bpy
import json
import os

def setup_lighting():
    # Delete default lights
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete(use_global=False)
    
    bpy.ops.object.light_add(type='SUN', radius=1, location=(30, 30, 5))
    key_light = bpy.context.object
    key_light.data.energy = 1.0
    
    bpy.ops.object.light_add(type='SUN', radius=1, location=(30, -30, 5))
    fill_light = bpy.context.object
    fill_light.data.energy = 1.0
    
    bpy.ops.object.light_add(type='SUN', radius=1, location=(-30, -30, 5))
    back_light = bpy.context.object
    back_light.data.energy = 1.0

    bpy.ops.object.light_add(type='SUN', radius=1, location=(-30, 30, 5))
    back_light = bpy.context.object
    back_light.data.energy = 1.0

    bpy.ops.object.light_add(type='SUN', radius=50, location=(0, 0, 10))
    back_light = bpy.context.object
    back_light.data.energy = 10.0


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


# Load camera parameters from JSON file
parameters_file = "camera_parameters.json"
additional_parameters_file = "camera_additional_parameters.json"
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

setup_lighting()

# Save the scene
output_dir = os.getcwd()
output_filepath = os.path.join(output_dir, "blender/untitled.blend")
bpy.ops.wm.save_as_mainfile(filepath=output_filepath)
