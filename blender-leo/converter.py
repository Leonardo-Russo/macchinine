from scipy.io import loadmat
import numpy as np
import json

def mat_to_json(mat_filepath='renderdata.mat', json_filepath='renderdata.json'):

    data = loadmat(mat_filepath)
    state = np.array(data['renderdata'])
    
    # Prepare data for .json
    data_to_save = []
    for row in state:
        position = row[:3].tolist()  # First three elements are XYZ position
        quaternion = row[3:].tolist()  # Last four elements are quaternion XYZW
        data_to_save.append({'position': position, 'quaternion': quaternion})
    
    # Write to JSON
    with open(json_filepath, 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)

    print(f'Converted {mat_filepath} to {json_filepath}')

mat_to_json()  # Call the function
