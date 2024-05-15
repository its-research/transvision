import json

a = [8,19,10,3,11,14,25,15,9,17,12,23,18,24]

# Read the JSON file
with open('/Users/libin/Desktop/020183.json', 'r') as file:
    data = json.load(file)

# Filter the elements based on indices in list 'a'
filtered_data = [data[i] for i in a if i < len(data)]

# Write the filtered elements to another file
with open('/Users/libin/Desktop/201832.json', 'w') as file:
    json.dump(filtered_data, file)

# Update the original file with remaining elements
remaining_data = [data[i] for i in range(len(data)) if i not in a]
with open('/Users/libin/Desktop/020183.json', 'w') as file:
    json.dump(remaining_data, file)