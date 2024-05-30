import bpy
import json

def import_stl(filepath, object_name="Spacecraft"):
    """
    Imports an STL file and renames the imported object.
    If the object already exists, it deletes the existing object first.
    """
    if object_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[object_name], do_unlink=True)
    
    bpy.ops.import_mesh.stl(filepath=filepath)
    imported_object = bpy.context.selected_objects[0]
    imported_object.name = object_name
    return bpy.data.objects[object_name]


def load_processed_data(filepath='test.json'):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def setup_camera(target, mode='PERSP', ortho_scale=15, distance=30):
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.name = "Camera"
    else:
        camera = bpy.data.objects["Camera"]
    
    # Set camera type
    camera.data.type = mode  # 'PERSP' for perspective, 'ORTHO' for orthographic
    
    if mode == 'ORTHO':
        camera.data.ortho_scale = ortho_scale  # Only relevant for orthographic mode
    
    # Add a Track To constraint
    if camera.constraints.get('Track To') is None:
        track_to = camera.constraints.new(type='TRACK_TO')
    else:
        track_to = camera.constraints['Track To']
    
    track_to.target = target
    track_to.track_axis = 'TRACK_NEGATIVE_Z'
    track_to.up_axis = 'UP_Y'
    
    # Optionally set an initial distance from the target for the camera
    camera.location = (target.location.x - distance, target.location.y + distance, target.location.z + distance)
    
    bpy.context.scene.camera = camera
    return camera




def apply_material_to_object(obj, color=(0.8, 0.8, 0.8, 1)):
    if not obj.data.materials:
        # Create a new material
        mat = bpy.data.materials.new(name=f"{obj.name}Material")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs['Base Color'].default_value = color  # Set color
        obj.data.materials.append(mat)


def create_and_animate_object(data, object_name="Spacecraft", stl_filepath=r"C:\Users\Leonardo\Desktop\Rendering\test.stl"):
    
    # Path to the STL file you want to import
    # stl_filepath = r"C:\Users\Leonardo\Desktop\Rendering\dragon3.stl"  # Update this path
    spacecraft = import_stl(stl_filepath, object_name=object_name)
    
    bpy.context.view_layer.objects.active = spacecraft
    spacecraft.rotation_mode = 'QUATERNION'

    # Setup the camera after the spacecraft is created
    spacecraft = bpy.data.objects[object_name]
    camera = setup_camera(spacecraft)

    # Adding sphere at the center
    bpy.ops.mesh.primitive_uv_sphere_add(radius=5, location=(0, 0, 0))
    central_sphere = bpy.context.object
    central_sphere.name = "CentralSphere"
    apply_material_to_object(central_sphere, color=(0.5, 0.5, 0.5, 1))  # Lighter gray for contrast

    
    # Animate the camera to follow the spacecraft
    frame_number = 1
    for frame_data in data:
        bpy.context.scene.frame_set(frame_number)
        position = frame_data['position']
        quaternion = frame_data['quaternion']
        
        print(position)
        print(quaternion)
        
        spacecraft.location = position
        spacecraft.rotation_quaternion = quaternion
        
        spacecraft.keyframe_insert(data_path="location", frame=frame_number)
        spacecraft.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)

        # Set the camera position relative to the spacecraft to maintain a fixed offset
        # This creates an isometric-like effect by keeping the camera at a fixed diagonal distance
        camera_dist = 30
        camera.location = (position[0] - camera_dist, position[1] + camera_dist, position[2] + camera_dist)  # Adjust the offset as needed
        
        camera.keyframe_insert(data_path="location", frame=frame_number)
        
        frame_number += 1  # Increment frame number


def setup_lighting():
    # Delete default lights
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete(use_global=False)
    
    # Key Light
    bpy.ops.object.light_add(type='SUN', radius=1, location=(10, -10, 10))
    key_light = bpy.context.object
    key_light.data.energy = 1.0
    
    # Fill Light
    bpy.ops.object.light_add(type='SUN', radius=1, location=(-10, 10, 5))
    fill_light = bpy.context.object
    fill_light.data.energy = 0.5
    
    # Back Light
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0, -10, -10))
    back_light = bpy.context.object
    back_light.data.energy = 0.7


def apply_material_to_spacecraft(object_name="Spacecraft", color=(0.8, 0.8, 0.8, 1)):
    # Adjusted to a more generic function to apply material to any object
    apply_material_to_object(bpy.data.objects[object_name], color)


def set_background_color(color=(0.05, 0.05, 0.05, 1)):  # Dark gray background
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = color


# Load the data and animate
data = load_processed_data(filepath=r"H:\My Drive\Codes\Python\Blender-Rendering\renderdata.json")
create_and_animate_object(data, stl_filepath=r"H:\My Drive\Codes\Python\Blender-Rendering\dragon2.stl")
setup_lighting()
apply_material_to_spacecraft()
set_background_color()