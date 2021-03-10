import tifffile as tiff, numpy as np, os, shutil, cv2, json, datetime, sys
from Datapair import get_dplist_from_dir
from tqdm import tqdm


data_dir =  '/home/chadrick/prj/kaggle/hubmap/teststuff/testoutput/refile_tiff_and_annot/210309_235006'

imgdir = os.path.join(data_dir, 'images')
annotdir = os.path.join(data_dir, 'annots')
dplist = get_dplist_from_dir(imgdir, annotdir)

print(len(dplist))



timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'testoutput/crop_targets/{timestamp}'

os.makedirs(outputdir)


saveimgdir = os.path.join(outputdir, 'images')
saveannotdir = os.path.join(outputdir, 'annots')
savedebugdrawdir = os.path.join(outputdir, 'debug_draw')

os.makedirs(saveimgdir)
os.makedirs(saveannotdir)
os.makedirs(savedebugdrawdir)


for dp in tqdm(dplist):

    bn = os.path.basename(dp.imgpath)
    fn, _ = os.path.splitext(bn)

    img = tiff.imread(dp.imgpath)

    if len(img.shape)==5:
        img = img[0,0,:]
        img = np.transpose(img, (1,2,0))    
    elif len(img.shape)==3: 
        pass
    else:
        raise Exception(f'invalid img shape: {img.shape}')


    print(f'img shape: {img.shape}')


        
    full_img_h = img.shape[0]
    full_img_w = img.shape[1]


    # filepath = '/home/chadrick/prj/kaggle/hubmap/data/train/1e2425f28.json'

    with open(dp.annotpath, 'r') as fd:
        readjson = json.load(fd)

    # print(len(readjson))
    # print(readjson[0])



    poly_coord_list_list = []

    for a in readjson:
        coord_list = a['geometry']['coordinates'][0]

        poly_coord_list_list.append(coord_list)

    # print(poly_coord_list_list)


    poly_bbox_coord_list = []

    for l in poly_coord_list_list:
        arr = np.array(l)

        x1 = np.min(arr[:,0])
        x2 = np.max(arr[:,0])
        y1 = np.min(arr[:,1])
        y2 = np.max(arr[:,1])

        poly_bbox_coord_list.append([x1,y1,x2,y2])

    print(f'poly count: {len(poly_bbox_coord_list)}')





    for i, (x1,y1,x2,y2) in enumerate(poly_bbox_coord_list):


        poly_coord_list = poly_coord_list_list[i]
        # print(f'poly_coord_list: {poly_coord_list}')

        crop = img[y1:y2, x1:x2]

        savepath = os.path.join(saveimgdir, f'{fn}_{i}.png')

        cv2.imwrite(savepath, crop)

        new_poly_coord_list = []

        for x,y in poly_coord_list:
            x = x - x1
            y = y - y1

            x = int(x)
            y = int(y)

            new_poly_coord_list.append([x,y])

        # draw line on crop

        copyimg = crop.copy()

        for j in range(len(new_poly_coord_list)-1):
            start = new_poly_coord_list[j]
            end = new_poly_coord_list[j+1]

            cv2.line(copyimg, tuple(start), tuple(end), (0,0,255),1)
        
        savepath = os.path.join(savedebugdrawdir, f'{fn}_{i}.png')
        cv2.imwrite(savepath, copyimg)


        
        savejson = {
            'poly_coord_list': new_poly_coord_list,
            'crop_img_w': copyimg.shape[1],
            'crop_img_h': copyimg.shape[0],
            'full_img_w': full_img_w,
            'full_img_h': full_img_h,
            'filename': fn
        }

        savepath = os.path.join(saveannotdir, f'{fn}_{i}.json')

        with open(savepath, 'w') as fd:
            json.dump(savejson, fd, indent=4, ensure_ascii=False)
        


            



# a = tiff.imread('/home/chadrick/prj/kaggle/hubmap/data/train/1e2425f28.tiff')

# print(a.shape)

# b= a
# print(b.shape)

# # print(b)

# b = a[0,0,:]

# print(b.shape)

# b = np.transpose(b, (1,2,0))

# print(b.shape)
