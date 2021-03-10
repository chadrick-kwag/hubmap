import os, glob, datetime, shutil



data_dir = '/home/chadrick/prj/kaggle/hubmap/data/train'


tiff_files = glob.glob(os.path.join(data_dir, '*.tiff'))


tuple_list = []

for f in tiff_files:
    bn = os.path.basename(f)
    fn, _ = os.path.splitext(bn)

    
    jsonfile = os.path.join(data_dir, f'{fn}.json')
    structure_file = os.path.join(data_dir, f'{fn}-anatomical-structure.json')

    assert os.path.exists(jsonfile)
    assert os.path.exists(structure_file)

    tuple_list.append([f, jsonfile, structure_file])


print(len(tuple_list))


timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'testoutput/refile_tiff_and_annot/{timestamp}'

os.makedirs(outputdir)


saveimgdir = os.path.join(outputdir, 'images')
saveannotdir = os.path.join(outputdir, 'annots')
save_structure_annotdir = os.path.join(outputdir, 'structure')

os.makedirs(saveimgdir)
os.makedirs(saveannotdir)
os.makedirs(save_structure_annotdir)


for a,b,c in tuple_list:

    bn = os.path.basename(a)

    savepath = os.path.join(saveimgdir, bn)
    shutil.copy2(a, savepath)


    bn = os.path.basename(b)

    savepath = os.path.join(saveannotdir, bn)
    shutil.copy2(b, savepath)


    bn = os.path.basename(c)

    savepath = os.path.join(save_structure_annotdir, bn)
    shutil.copy2(c, savepath)


