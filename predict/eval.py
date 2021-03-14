import torch, yaml, os, datetime, argparse, sys, glob, cv2, json, numpy as np, shutil
from tqdm import tqdm

sys.path.append(os.path.abspath(".."))

from model.model_map import model_map
from dataprovider.dataset import EvalDataset
from criterion.dice import batch_dice



def draw_overlay(orig_img, mask, color = [0,0,255]):

    overlay_mask = mask > 0

    mask_img = np.ones_like(orig_img)

    mask_img = mask_img * np.array(color)

    # print(f'mask_img shp : {mask_img.shape}')

    adjusted_ratio_mask = 0.6 * overlay_mask.astype(float)

    new_img = orig_img.astype(float) * (1-adjusted_ratio_mask) + mask_img.astype(float) * adjusted_ratio_mask
    new_img = new_img.astype('uint8')
    

    return new_img



def main(config):

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
    assert os.path.isdir(target)



    batch_size = config['batch_size']

    resize = (config['resize_w'], config['resize_h'])

    dataset = EvalDataset(target, resize)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)


    timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    outputdir = f'testoutput/eval/{timestamp}'

    os.makedirs(outputdir)


    # copy used config

    savepath = os.path.join(outputdir, 'usedconfig.yaml')

    with open(savepath, 'w') as fd:
        yaml.dump(config, fd)

    threshold = config['threshold']


    dice_result = {}
    dice_list = []

    for img_data, mask_data, imgpath_list, orig_img_data in tqdm(dataloader):



        img_data = img_data.to(torch.float).permute(0,3,1,2).to(device)
        mask_nparr = mask_data.numpy()

        orig_img_nparr = orig_img_data.numpy().astype('uint8')



        with torch.no_grad():

            pred = net(img_data)

        # print(f'pred: {pred}')

        pred_nparr = pred.cpu().detach().numpy()

        del pred
        torch.cuda.empty_cache()

        # print(pred_nparr.shape)


        pred_mask = pred_nparr > threshold

        maskimg = pred_mask.astype('uint8') * 255

        dice_list = batch_dice(~pred_mask, ~mask_nparr)

        for i,(m, gt_m, d, orig_img) in enumerate(zip(maskimg, mask_nparr, dice_list, orig_img_nparr)):

            imgpath = imgpath_list[i]

            overlay_img = draw_overlay(orig_img, m)
            # print(f'overlay_img: {overlay_img}')

            overlay_img = draw_overlay(overlay_img, gt_m, color=[255,0,0])

            bn = os.path.basename(imgpath)
            fn, _  = os.path.splitext(bn)

            savepath = os.path.join(outputdir, f'{fn}.png')

            cv2.imwrite(savepath, overlay_img)

            dice_result[bn] = d
            dice_list.append(d)


    mean_dice = sum(dice_list) / len(dice_list)

    print(f'bg mean_dice: {mean_dice}')

    savejson = {
        'mean_dice': mean_dice,
        "results": dice_result
    }

    savepath = os.path.join(outputdir, 'result.json')

    with open(savepath, 'w') as fd:
        json.dump(savejson, fd, indent=4, ensure_ascii=False)

    return savejson, os.path.abspath(outputdir)


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str, help='config file')

    args = parser.parse_args()

    assert os.path.exists(args.config)

    with open(args.config, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    main(config)