import argparse, os, sys, torch, tqdm, yaml
from torch import optim

sys.path.append(os.path.abspath(".."))

from dataprovider.dataset import Dataset
from model.model_map import model_map
from loss.loss_map import loss_map


parser = argparse.ArgumentParser()

parser.add_argument('config', type=str, help='config file')

args = parser.parse_args()


with open(args.config, 'r') as fd:
    config = yaml.load(fd, Loader=yaml.FullLoader)


""" setup device """

gpu = config.get('gpu', None)

if gpu=='cpu' or gpu is None:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{gpu}')


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


epochs = config['epochs']



for epoch_i in range(epochs):

    for step, batch_data in enumerate(dataloader):

        # print(batch_data)

        optimizer.zero_grad()
        
        batch_img_data, batch_mask_data = batch_data

        # print(batch_data)







        batch_img_data = batch_img_data.float().to(device)

        batch_img_data = batch_img_data.permute(0,3,1,2)


        print(f'batch_img_data shp: {batch_img_data.shape}')
        print(f'batch_img_data device: {batch_img_data.device}')

        pred_output = net(batch_img_data)

        print(f'pred_output shp: {pred_output.shape}')

        # calc loss

        loss = loss_fn(pred_output, batch_mask_data)

        print(f'loss={loss.item()}')

        loss.backward()

        optimizer.step()



    

        
        

