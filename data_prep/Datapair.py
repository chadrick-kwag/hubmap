import glob, os

class DataPair:

    def __init__(self, imgpath, annotpath):
        self.imgpath = imgpath
        self.annotpath = annotpath

    def is_complete(self):

        if self.imgpath is not None and self.annotpath is not None:
            return True
        
        return False



def get_dplist_from_dir(imgdir, annotdir):

    assert os.path.exists(imgdir)
    assert os.path.exists(annotdir)

    imgfiles = glob.glob(os.path.join(imgdir, '*'))
    annotfiles = glob.glob(os.path.join(annotdir, '*'))

    name_to_dp_dict = {}

    for f in imgfiles:
        bn = os.path.basename(f)

        fn, _ = os.path.splitext(bn)

        name_to_dp_dict[fn] = DataPair(f, None)

    
    for f in annotfiles:
        bn = os.path.basename(f)

        fn, _ = os.path.splitext(bn)

        dp = name_to_dp_dict.get(fn, None)

        if dp is None:
            continue
    
        dp.annotpath = f
    

    complete_dp_list = [a for a in name_to_dp_dict.values() if a.is_complete()]

    return complete_dp_list
