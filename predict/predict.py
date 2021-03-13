import torch, yaml, os, datetime, argparse, sys, glob, cv2
from tqdm import tqdm

sys.path.append(os.path.abspath(".."))

from model.model_map import model_map
from dataprovider.dataset import PredictDataset


parser = argparse.ArgumentParser()

parser.add_argument('config', type=str, help='config file')

args = parser.parse_args()

assert os.path.exists(args.config)

with open(args.config, 'r') as fd:
    config = yaml.load(fd, Loader=yaml.FullLoader)


model_type = config['model_type']

model_builder = model_map[model_type]

net = model_builder(3)

net.load_state_dict(torch.load(config['ckpt']))


if config['gpu'] is None or config['gpu'] == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{config["gpu"]}')


net = net.to(device)



target = config['target']

assert os.path.exists(target)

if os.path.isfile(target):
    files = [target]
elif os.path.isdir(target):

    files = glob.glob(os.path.join(target, '*'))
else:
    raise Exception('invalid target type')

assert len(files) > 0 


batch_size = config['batch_size']

resize = (config['resize_w'], config['resize_h'])

dataset = PredictDataset(files, resize)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)


timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'testoutput/predict/{timestamp}'

os.makedirs(outputdir)

threshold = 0.5


for img_data, imgpath_list in tqdm(dataloader):

    # print(f'imgpath_list: {imgpath_list}')

    # print(f'img_data: {img_data}')


    img_data = img_data.to(torch.float).permute(0,3,1,2).to(device)



    with torch.no_grad():

        pred = net(img_data)

    # print(f'pred: {pred}')

    pred_nparr = pred.cpu().detach().numpy()

    del pred
    torch.cuda.empty_cache()

    # print(pred_nparr.shape)


    mask = pred_nparr > threshold

    maskimg = mask.astype('uint8') * 255

    for i,m in enumerate(maskimg):

        imgpath = imgpath_list[i]

        bn = os.path.basename(imgpath)
        fn, _  = os.path.splitext(bn)

        savepath = os.path.join(outputdir, f'{fn}.png')

        cv2.imwrite(savepath, m)

