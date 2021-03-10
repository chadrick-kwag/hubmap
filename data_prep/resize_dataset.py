import tifffile as tiff, os, cv2, shutil, numpy as np, json, datetime
from Datapair import get_dplist_from_dir



data_dir = '/home/chadrick/prj/kaggle/hubmap/data_prep/testoutput/refile_tiff_and_annot/210309_235006'


class cfg:

    resize_w = 2000
    resize_h = 2000



imgdir = os.path.join(data_dir, 'images')
annotdir = os.path.join(data_dir, 'annots')
structuredir = os.path.join(data_dir, 'structure')

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


img_h = img.shape[0]
img_w = img.shape[1]

resized_img = cv2.resize(img, (cfg.resize_w, cfg.resize_h))

w_factor = cfg.resize_w / img_w
h_factor = cfg.resize_h / img_h


with open(dp.annotpath, 'r') as fd:
    readjson = json.load(fd)

elem_list = readjson

print(f'elem size: {len(elem_list)}')

poly_list = [a['geometry']['coordinates'][0] for a in elem_list]

# print(poly_list)
print(len(poly_list))

new_poly_list = []

for p in poly_list:
    new_p = []

    for x,y in p:
        x = int(x * w_factor)
        y = int(y * h_factor)

        new_p.append([x,y])

    new_poly_list.append(new_p)






timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'testoutput/resize_dataset/{timestamp}'

os.makedirs(outputdir)


savepath = os.path.join(outputdir, 'resized_img.png')

cv2.imwrite(savepath, resized_img)


savejson = {
    'poly_coord': new_poly_list
}

savepath = os.path.join(outputdir, 'resized_annot.json')

with open(savepath, 'w') as fd:
    json.dump(savejson, fd, indent=4, ensure_ascii=False)



