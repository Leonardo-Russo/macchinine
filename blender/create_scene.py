import bpy
import sys
import os

# Reset Blender to default settings to create a new file
bpy.ops.wm.read_factory_settings(use_empty=True)

output_dir = os.getcwd()  # Default to current directory if not provided

# Specify the output file path
output_filepath = os.path.join(output_dir, "blender/untitled.blend")

# Save the scene
bpy.ops.wm.save_as_mainfile(filepath=output_filepath)
