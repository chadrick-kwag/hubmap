import os, tifffile as tiff, sys, json, numpy as np, datetime, cv2

from Datapair import get_dplist_from_dir
from tqdm import tqdm


data_dir = '/home/chadrick/prj/kaggle/hubmap/data_prep/testoutput/refile_tiff_and_annot/210309_235006'


imgdir = os.path.join(data_dir, 'images')
annotdir = os.path.join(data_dir, 'annots')

dplist = get_dplist_from_dir(imgdir, annotdir)

dp = dplist[0]




img = tiff.imread(dp.imgpath)

if len(img.shape)==5:
    img = img[0,0,:]
    img = np.transpose(img, (1,2,0))    
elif len(img.shape)==3: 
    pass
else:
    raise Exception(f'invalid img shape: {img.shape}')


img = tiff.imread(dp.imgpath)

if len(img.shape)==5:
    img = img[0,0,:]
    img = np.transpose(img, (1,2,0))    
elif len(img.shape)==3: 
    pass
else:
    raise Exception(f'invalid img shape: {img.shape}')


img_h = img.shape[0]
img_w = img.shape[1]


with open(dp.annotpath, 'r') as fd:
    readjson = json.load(fd)


poly_list = []

for elem in readjson:

    poly = elem['geometry']['coordinates'][0]

    poly_list.append(poly)

print(len(poly_list))


slice_w = 800
slice_h = 800

mask = np.zeros((img_h, img_w), dtype='uint8')

print(f'mask shape: {mask.shape}')

# poly_arr = np.array(poly_list)

# print(f'poly_arr shp: {poly_arr.shape}')


for poly in poly_list:
    poly_arr = np.array(poly)
    print(f'poly_arr shp: {poly_arr.shape}')
    cv2.fillPoly(mask, [poly_arr], 255)


print(f'mask unique values: {np.unique(mask)}')
mask_sum = np.sum(mask)

print(f"mask sum: {mask_sum}")


timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'testoutput/slice_dataset/{timestamp}'
os.makedirs(outputdir)

savepath = os.path.join(outputdir, 'mask.png')

cv2.imwrite(savepath, mask)


saveimgdir = os.path.join(outputdir, 'images')
saveannotdir = os.path.join(outputdir, 'annots')

os.makedirs(saveimgdir)
os.makedirs(saveannotdir)


slice_x_start_list=[]
slice_x_end_list = []


slice_w_num = img_w// slice_w


for i in range(slice_w_num):

    slice_x_start = slice_w * i
    slice_x_end = slice_w * (i+1)

    if slice_x_end > img_w:
        slice_x_end = img_w
        slice_x_start = slice_x_end - slice_w

    slice_x_start_list.append(slice_x_start)
    slice_x_end_list.append(slice_x_end)


slice_y_start_list = []
slice_y_end_list = []

slice_h_num = img_h // slice_h

for i in range(slice_h_num):

    slice_y_start = slice_h * i
    slice_y_end = slice_h * (i+1)

    if slice_y_end > img_h:
        slice_y_end = img_h
        slice_y_start = slice_y_end - slice_h
    
    slice_y_start_list.append(slice_y_start)
    slice_y_end_list.append(slice_y_end)


print(f'slice w num: {slice_w_num}, slice h num: {slice_h_num}')


gen_count = 0

progress = tqdm(total=len(slice_x_start_list) * len(slice_y_start_list))

for slice_x_start, slice_x_end in zip(slice_x_start_list, slice_x_end_list):

    for slice_y_start, slice_y_end in zip(slice_y_start_list, slice_y_end_list):

        img_crop = img[slice_y_start: slice_y_end, slice_x_start:slice_x_end]

        mask_crop = mask[slice_y_start: slice_y_end, slice_x_start: slice_x_end]
    

        savepath = os.path.join(saveimgdir, f'{gen_count}.png')
        cv2.imwrite(savepath, img_crop)


        savepath = os.path.join(saveannotdir, f'{gen_count}.png')
        cv2.imwrite(savepath, mask_crop)

        gen_count +=1

        progress.update(1)
    




