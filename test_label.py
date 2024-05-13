import os
import json

folder_path = '/Users/libin/Desktop/data/cooperative-vehicle-infrastructure/cooperative/label_world'  # Replace with the actual folder path

# Iterate over all files in the folder
filenames =[]
nums = []
veh_path = '/Users/libin/Desktop/data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side/label/lidar/'

max_diff =0
max_file = ''
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Check if the file is a JSON file
    if filename.endswith('.json'):
        with open(file_path, 'r') as file:
            data = json.load(file)
            print(data)
            
            num_elements = len(data)
            print(f"File: {filename}, Number of Elements: {num_elements}")
            exit()
            
        veh_file_path = veh_path + filename
        
        with open(file_path, 'r') as file:
            data = json.load(file)
            veh_num_elements = len(data)
            print(f"File: {filename}, Number of Elements: {num_elements}")
            
        diff = num_elements - veh_num_elements
        if max_diff < diff:
            max_diff = diff
            max_file = filename
print (max_diff, max_file)