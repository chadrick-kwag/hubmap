import torch, numpy as np
from criterion.dice import batch_dice


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

            _dice_list = batch_dice(~pred_mask, ~batch_mask_data)
            dice_list.extend(_dice_list)

        mean_dice = sum(dice_list) / len(dice_list)

        print(f'mean bg_dice: {mean_dice}')

        for a in self.dice_subscribers:
            a.run(epoch, step, global_step, mean_dice)

        if self.writer is not None:
            self.writer.add_scalar('valid/mean_bg_dice', mean_dice, global_step)
