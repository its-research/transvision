import pickle

file_path = "./data/nuscenes/nuscenes_infos_val_occ.pkl"

with open(file_path, "rb") as file:
    data = pickle.load(file)

for item in data['infos']:
    print(item)
    break