import os, random, shutil, datetime, json
from Datapair import get_dplist_from_dir

random.seed()

# data_dir = '/home/chadrick/prj/kaggle/hubmap/data_prep/testoutput/select_save_data/210311_231110/mask_coverage_satisfied'
data_dir = '/home/chadrick/prj/kaggle/hubmap/data_prep/testoutput/select_save_data/210311_231110/no_mask'

imgdir = os.path.join(data_dir, 'images')
annotdir = os.path.join(data_dir, 'annots')

dplist = get_dplist_from_dir(imgdir, annotdir)


print(len(dplist))



sample_size = 50

assert len(dplist) > sample_size

sample = random.sample(dplist, sample_size)


timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'testoutput/sample_data/{timestamp}'

os.makedirs(outputdir)


savejson = {
    'data_dir': data_dir,
    'sample_size': sample_size
}

savepath = os.path.join(outputdir, 'info.json')

with open(savepath, 'w') as fd:
    json.dump(savejson, fd, indent=4, ensure_ascii=False)



def save_dp_list(savedir, dp_list):

    imgdir = os.path.join(savedir, 'images')
    annotdir = os.path.join(savedir, 'annots')

    os.makedirs(imgdir)
    os.makedirs(annotdir)

    for dp in dp_list:

        bn = os.path.basename(dp.imgpath)
        savepath = os.path.join(imgdir, bn)

        shutil.copy2(dp.imgpath, savepath)

        bn = os.path.basename(dp.annotpath)
        savepath = os.path.join(annotdir, bn)

        shutil.copy2(dp.annotpath, savepath)

save_dp_list(outputdir, sample)

