import os, sys, cv2, json, matplotlib.pyplot as plt, numpy as np, datetime, glob
import tifffile as tiff


data_dir = '/home/chadrick/prj/kaggle/hubmap/teststuff/testoutput/refile_tiff_and_annot/210309_235006/images'

files = glob.glob(os.path.join(data_dir, '*.tiff'))

print(len(files))

img_size_list = []

name_to_img_shp_dict = {}

for f in files:
    img = tiff.imread(f)

    if len(img.shape)==5:
        img = img[0,0,:]
        img = np.transpose(img, (1,2,0))    
    elif len(img.shape)==3: 
        pass
    else:
        raise Exception(f'invalid img shape: {img.shape}')

    img_h = img.shape[0]
    img_w = img.shape[1]

    img_size_list.append([img_w, img_h])

    bn = os.path.basename(f)
    fn, _= os.path.splitext(bn)

    name_to_img_shp_dict[fn] = {
        'img_w': img_w,
        'img_h': img_h
    }


img_w_arr = np.array([a[0] for a in img_size_list])
img_h_arr = np.array([a[1] for a in img_size_list])



figure = plt.figure()


plt.hist2d(img_w_arr, img_h_arr)
plt.colorbar()


timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'testoutput/tiff_image_stat/{timestamp}'

os.makedirs(outputdir)

print(f'outputdir: {outputdir}')


savepath = os.path.join(outputdir, 'test.png')

figure.savefig(savepath)



savepath = os.path.join(outputdir, 'savejson.json')

with open(savepath, 'w') as fd:
    json.dump(name_to_img_shp_dict, fd, indent=4, ensure_ascii=False)
