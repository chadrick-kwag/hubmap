import os, tifffile as tiff, sys, json, numpy as np, datetime, cv2, rasterio

from Datapair import get_dplist_from_dir
from tqdm import tqdm

from rasterio.windows import Window





def get_grid_coords(img_w, img_h, slice_w, slice_h):

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

    grid_coord_list =[]

    for slice_x_start, slice_x_end in zip(slice_x_start_list, slice_x_end_list):
        for slice_y_start, slice_y_end in zip(slice_y_start_list, slice_y_end_list):

            grid_coord_list.append([slice_x_start, slice_y_start, slice_x_end, slice_y_end])
    
    return grid_coord_list

data_dir = '/home/chadrick/prj/kaggle/hubmap/data_prep/testoutput/refile_tiff_and_annot/210313_113841'


imgdir = os.path.join(data_dir, 'images')
annotdir = os.path.join(data_dir, 'annots')

dplist = get_dplist_from_dir(imgdir, annotdir)

dp = dplist[0]



timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'testoutput/slice_dataset_v2/{timestamp}'

os.makedirs(outputdir)

saveimgdir = os.path.join(outputdir, 'images')
saveannotdir = os.path.join(outputdir, 'annots')

os.makedirs(saveimgdir)
os.makedirs(saveannotdir)

gen_count = 0

for dp_i, dp in enumerate(dplist):

    try:

        print(f'working on {dp.imgpath} , {dp_i+1}/{len(dplist)}')

        with rasterio.open(dp.imgpath) as src:
            # print(f'raster shp: {src.shape}')
            # print(src.count)
            # print(src.width)
            # print(src.height)

            img_w = src.width
            img_h = src.height

            mask = np.zeros((img_h, img_w), dtype='uint8')

            print(f'mask shp: {mask.shape}')



            with open(dp.annotpath, 'r') as fd:
                readjson = json.load(fd)


            poly_list = []

            for elem in readjson:

                poly = elem['geometry']['coordinates'][0]

                # _poly = [[a[1], a[0]] for a in poly]

                poly_list.append(poly)

            # print(len(poly_list))



            # poly_arr = np.array(poly_list)

            # print(f'poly_arr shp: {poly_arr.shape}')


            for poly in poly_list:
                poly_arr = np.array(poly)
                poly_arr = poly_arr.astype(int)
                # print(f'poly_arr shp: {poly_arr.shape}')
                # print(poly_arr)
                cv2.fillPoly(mask, [poly_arr], 255)


            channels = [1,2,3] if src.count == 3 else [1,1,1]


            slice_w = 800
            slice_h = 800

            grid_coord_list = get_grid_coords(img_w, img_h, slice_w, slice_h)

            for x1,y1,x2,y2 in tqdm(grid_coord_list):
                image = src.read(channels, window = Window(x1,y1, x2-x1, y2-y1))

                # print(image.shape)
                image = np.transpose(image, (1,2,0))

                mask_crop = mask[y1:y2, x1:x2]

                savepath = os.path.join(saveimgdir, f'{gen_count}.png')
                cv2.imwrite(savepath, image)

                savepath = os.path.join(saveannotdir, f'{gen_count}.png')
                cv2.imwrite(savepath, mask_crop)

                gen_count+=1
                

        # break

    except Exception as e:
        print(f'error on {dp.imgpath}')
        raise e


