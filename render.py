import subprocess
import os

# Specify path to Blender executable and project file
blender = r"D:\Programs\Blender 4.0\blender.exe"
blend_path = r"G:\Il mio Drive\Codes\Python\Macchinine\blender\untitled.blend"

# Create a new Blender project
create_command = [blender, '-b', '-P', 'blender/create_scene.py']
create_process = subprocess.run(create_command, capture_output=False, text=True)

# Load the Scene in the Blender Project
load_command = [blender, '-b', blend_path, '-P', 'blender/load_scene.py']
load_process = subprocess.run(load_command, capture_output=False, text=True)

# Render the Blender project
render_command = [blender, '-b', blend_path, '-P', 'blender/render_scene.py']
render_process = subprocess.run(render_command, capture_output=False, text=True)

# Check for errors and print output if necessary
if create_process.returncode != 0:
    print("Error creating Blender project:")
    print(create_process.stderr)
if load_process.returncode != 0:
    print("Error loading camera into Blender project:")
    print(load_process.stderr)
if render_process.returncode != 0:
    print("Error rendering Blender project:")
    print(render_process.stderr)

# else:
#     # remove the blender project files
#     os.remove("blender/untitled.blend")
#     os.remove("blender/untitled.blend1")