import torch


def logit_sanitation(val, min_val):
    unsqueezed_a = torch.unsqueeze(val, -1)
    limit = torch.ones_like(unsqueezed_a) * min_val
    a = torch.cat((unsqueezed_a, limit),-1)
    values, _= torch.max(a,-1)
    return values
  
  
def manual_bce_loss(pred_tensor, gt_tensor, epsilon = 1e-8):
    a = logit_sanitation(1-pred_tensor, epsilon)
    b = logit_sanitation(pred_tensor, epsilon)
    loss = - ( (1- gt_tensor) * torch.log(a) + gt_tensor * torch.log(b))
    return loss


def manual_focal_loss(pred_tensor, gt_tensor, gamma, epsilon = 1e-8):

    # print(f'pred tensor: {pred_tensor.shape}, gt_tensor: {gt_tensor.shape}')

    a = logit_sanitation(1-pred_tensor, epsilon)
    b = logit_sanitation(pred_tensor, epsilon)

    logit = (1-gt_tensor) * a + gt_tensor * b
    focal_loss = - (1-logit) ** gamma * torch.log(logit)
    return focal_loss


def focal_loss_on_cpu(pred_tensor, gt_tensor, gamma, epsilon = 1e-8):


    pred_tensor = pred_tensor.cpu()
    gt_tensor = gt_tensor.cpu()

    loss =  manual_focal_loss(pred_tensor, gt_tensor, gamma=gamma, epsilon=epsilon)

    # print(f'loss shp: {loss.shape}')

    return loss.mean()


def dice_loss(pred_tensor, gt_tensor, epsilon=1e-8):

    pred_tensor = pred_tensor.cpu()
    gt_tensor = gt_tensor.cpu()

    # pred_flat = pred_tensor.view(-1)
    # gt_flat = gt_tensor.view(-1)

    intersection = pred_tensor * gt_tensor

    dice = 1-(2 * intersection.sum() ) / (pred_tensor.sum() + gt_tensor.sum() + epsilon)

    return dice

