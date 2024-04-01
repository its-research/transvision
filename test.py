import gzip
import pickle

# import numpy as np

d1 = pickle.load(open('submission1', 'rb'))
d2 = pickle.load(open('submission2', 'rb'))

result_dict = {}

for token in d1['results']:
    # print(d1['results'][token])
    result_dict.update({token: d1['results'][token]})

for token in d2['results']:
    result_dict.update({token: d2['results'][token]})

final_submission_dict = {
    'method': 'occ',
    'team': 'The16thRoute',
    'authors': 'Bin Li',
    'e-mail': 'lbin@outlook.com',
    'institution / company': 'SEU',
    'country / region': 'China/Beijing',
    'results': result_dict,
}
with open('submission.gz', 'wb') as f:
    f.write(gzip.compress(pickle.dumps(final_submission_dict), mtime=0))

# with open(file_path, 'rb') as file:
#     data = pickle.load(file)
# for item in data['results']:
#     print(item)

# print(len(data['results']))
# print(data['results'])
# print(len(data['results']['557161e4e2374f42bac36c36b93a97fd']['pcd_flow']))

# label_file_path = 'labels.npz'
# label_data = np.load(label_file_path)

# for key in label_data.keys():
#     print(f"Key: {key}")
#     print(f"Value: {(label_data[key][0][1])}")
