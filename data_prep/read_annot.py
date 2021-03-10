import json, numpy as np


filepath = '/home/chadrick/prj/kaggle/hubmap/data/train/1e2425f28.json'

with open(filepath, 'r') as fd:
    readjson = json.load(fd)

print(len(readjson))
print(readjson[0])



poly_coord_list_list = []

for a in readjson:
    coord_list = a['geometry']['coordinates']

    poly_coord_list_list.append(coord_list)

print(poly_coord_list_list)


poly_bbox_coord_list = []

for l in poly_coord_list_list:
    arr = np.array(l)

    x1 = np.min(arr[:,0])
    x2 = np.max(arr[:,0])
    y1 = np.min(arr[:,1])
    y2 = np.max(arr[:,1])

    poly_bbox_coord_list.append([x1,y1,x2,y2])

print(poly_bbox_coord_list)