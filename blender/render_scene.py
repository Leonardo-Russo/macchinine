import bpy
import os

# # Set render resolution
# bpy.context.scene.render.resolution_x = 1920  # Example for Full HD
# bpy.context.scene.render.resolution_y = 1080
# bpy.context.scene.render.resolution_percentage = 100

# Set render resolution
bpy.context.scene.render.resolution_x = 1280  # Example for Full HD
bpy.context.scene.render.resolution_y = 720
bpy.context.scene.render.resolution_percentage = 70

# Set render frame range
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 74  # Adjust based on your animation length

# Set output file format to FFmpeg video
bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
bpy.context.scene.render.ffmpeg.format = 'MPEG4'
bpy.context.scene.render.ffmpeg.codec = 'H264'
bpy.context.scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
bpy.context.scene.render.ffmpeg.ffmpeg_preset = 'GOOD'

# Set output file path
output_dir = os.getcwd()
output_filepath = os.path.join(output_dir, "scene.mp4")
bpy.context.scene.render.filepath = output_filepath

# Use this if you want to overwrite existing files
bpy.context.scene.render.use_overwrite = True

# Use this if you want Blender to create a new directory if it doesn't exist
bpy.context.scene.render.use_placeholder = True

# Optionally, set the number of render threads
# bpy.context.scene.render.threads_mode = 'FIXED'
# bpy.context.scene.render.threads = 4  # Adjust based on your machine's capability


# Render the animation
bpy.ops.render.render(animation=True)