import argparse, os, sys, torch, tqdm, yaml, datetime, shutil
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(".."))

from dataprovider.dataset import Dataset
from model.model_map import model_map
from loss.loss_map import loss_map
from callback.manualsavecallback import ManualSaveCallback
from callback.validcallback import ValidationCallback
from callback.monitor_metric_ckptsave_callback import MonitorCkptSaveCallback

parser = argparse.ArgumentParser()

parser.add_argument('config', type=str, help='config file')

args = parser.parse_args()


with open(args.config, 'r') as fd:
    config = yaml.load(fd, Loader=yaml.FullLoader)


""" setup savedir """

timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'ckpt/train_v1/{timestamp}'

os.makedirs(outputdir)


""" copy used config """

savepath = os.path.join(outputdir, 'usedconfig.yaml')
shutil.copy2(args.config, savepath)


""" setup device """

gpu = config.get('gpu', None)

if gpu=='cpu' or gpu is None:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{gpu}')

""" setup dataloader """

dataset_list = []

for d in config['train_data_dir']:
    print(d)
    ds = Dataset(d)
    dataset_list.append(ds)

# print(f'dataset_list: {dataset_list}')

if len(dataset_list)==1:
    dataset = dataset_list[0]
else:
    raise Exception('multiple datasets')

print(f'dataset len: {len(dataset)}')

print(f'batch size: {config["batch_size"]}')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

print(len(dataloader))


valid_dataset_list = []

for d in config['valid_data_dir']:
    ds = Dataset(d)
    valid_dataset_list.append(ds)

if len(valid_dataset_list)==1:
    dataset = valid_dataset_list[0]
else:
    raise Exception()

valid_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0)




""" setup model """

model_type = config.get('model_type', None)

if model_type is None:
    raise Exception('no model type defined in config')


model_builder = model_map.get(model_type, None)

if model_builder is None:
    raise Exception(f'invalid model type: {model_type}')


net = model_builder(3)
net = net.to(device)


""" setup optimizer """

optimizer = torch.optim.Adam(net.parameters(),lr=config['lr'])

""" setup loss fn """

loss_type = config['loss_type']
loss_fn = loss_map[loss_type]




""" setup writer """

logdir = os.path.join(outputdir, 'logs')
os.makedirs(logdir)
writer = SummaryWriter(log_dir=logdir)

""" setup callback """

savedir = os.path.join(outputdir, 'periodic_save')
os.makedirs(savedir)
manual_save_callback = ManualSaveCallback(savedir, net)


savedir = os.path.join(outputdir, 'valid_dice_save')
os.makedirs(savedir)
dice_save_callback = MonitorCkptSaveCallback(net, savedir, 'max', 'dice')


valid_callback = ValidationCallback(net, valid_dataloader, device, dice_subscribers=[dice_save_callback], writer=writer)





### setup train iter option

epochs = config['epochs']
manual_save_period = config['manual_save_period']
run_valid_period = config['run_valid_period']




global_step = 0

for epoch_i in range(epochs):

    for step, batch_data in enumerate(dataloader):

        global_step+=1

        # print(batch_data)

        optimizer.zero_grad()
        
        batch_img_data, batch_mask_data = batch_data

        # print(batch_data)



        batch_img_data = batch_img_data.float().to(device)

        batch_img_data = batch_img_data.permute(0,3,1,2)


        # print(f'batch_img_data shp: {batch_img_data.shape}')
        # print(f'batch_img_data device: {batch_img_data.device}')

        pred_output = net(batch_img_data)

        # print(f'pred_output shp: {pred_output.shape}')

        # calc loss

        loss = loss_fn(pred_output, batch_mask_data)

        print(f'epoch={epoch_i}, step={step}, loss={loss.item()}')
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.flush()

        loss.backward()

        optimizer.step()

        

        if global_step % manual_save_period == 0:

            manual_save_callback.run(epoch_i, step)

        if global_step % run_valid_period == 0:

            valid_callback.run(epoch_i, step, global_step)


       