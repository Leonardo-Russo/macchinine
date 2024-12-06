import subprocess
import os

# # Leo Laptop Setup
# blender = r"D:\Programs\Blender 4.0\blender.exe"
# blend_path = r"G:\Il mio Drive\Codes\Python\Macchinine\blender\untitled.blend"

# Leo DLR Setup
blender = r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"
blend_path = r"C:\Users\russ_le\Codes\macchinine\blender\untitled.blend"

# Load the pre-made .blend file in the Blender Project and read the csv data
load_command = [blender, '-b', blend_path, '-P', 'blender/load_scene.py']
load_process = subprocess.run(load_command, capture_output=False, text=True)

# Render the Blender project
render_command = [blender, '-b', blend_path, '-P', 'blender/render_scene.py']
render_process = subprocess.run(render_command, capture_output=False, text=True)

# Check for errors and print output if necessary
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