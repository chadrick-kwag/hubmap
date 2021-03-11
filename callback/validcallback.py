import torch, numpy as np


class ValidationCallback:

    def __init__(self, net, dataloader, device, threshold=0.5, dice_subscribers = [], writer=None) -> None:

        self.net = net
        self.dataloader = dataloader
        self.device = device
        self.threshold = threshold

        self.dice_subscribers = dice_subscribers

        self.writer = writer
    
    def run(self, epoch, step, global_step):


        dice_list = []

        for batch_data in self.dataloader:
            
            batch_img_data, batch_mask_data = batch_data

            
            batch_img_data = batch_img_data.float().to(self.device)

            batch_img_data = batch_img_data.permute(0,3,1,2)


            with torch.no_grad():

                pred = self.net(batch_img_data)

            # calculate dice coefficient

            pred_arr = pred.cpu().detach().numpy()

            del pred
            torch.cuda.empty_cache()

            pred_mask = pred_arr > self.threshold

            batch_mask_data = batch_mask_data.numpy()

            # print(f'batch_mask_data: {batch_mask_data.shape}')
            # print(f'batch_mask_data dtype: {batch_mask_data.dtype}')


            and_mask = batch_mask_data * pred_mask

            delimiter = batch_mask_data + pred_mask
            delimiter = delimiter.astype(bool)

            and_mask_flat = np.reshape(and_mask, [and_mask.shape[0], -1])
            delimiter_flat = np.reshape(delimiter, [delimiter.shape[0], -1])

            and_mask_sum_arr = np.sum(and_mask_flat, axis=1)
            delimiter_sum_arr = np.sum(delimiter_flat, axis=1)
            delimiter_sum_arr = delimiter_sum_arr.astype(float) + 1e-8 # avoid divide by zero

            # print(f'and_mask_sum_arr: {and_mask_sum_arr}, delimiter_sum_arr: {delimiter_sum_arr}')

            dice_arr = and_mask_sum_arr / delimiter_sum_arr

            dice_list.extend(dice_arr.tolist())

        # print(f'dice_list: {dice_list}')

        mean_dice = sum(dice_list) / len(dice_list)

        print(f'mean_dice: {mean_dice}')

        for a in self.dice_subscribers:
            a.run(epoch, step, global_step, mean_dice)

        if self.writer is not None:
            self.writer.add_scalar('valid/mean_dice', mean_dice, global_step)

        
