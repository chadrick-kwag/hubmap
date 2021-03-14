import numpy as np

def batch_dice(pred_arr, mask_arr, epsilon=1e-8):
    """

    pred_arr: bool nparr
    mask_arr: bool nparr
    """
    


    # print(f'batch_mask_data: {batch_mask_data.shape}')
    # print(f'batch_mask_data dtype: {batch_mask_data.dtype}')


    and_mask = mask_arr * pred_arr

    and_mask_flat = np.reshape(and_mask, [and_mask.shape[0], -1])
    and_mask_sum_arr = np.sum(and_mask_flat, axis=1)

    pred_flat = np.reshape(pred_arr, [pred_arr.shape[0], -1])
    mask_flat = np.reshape(mask_arr, [mask_arr.shape[0], -1])

    pred_sum = np.sum(pred_flat, axis=1)
    mask_sum = np.sum(mask_flat, axis=1)

    delimiter_arr = pred_sum + mask_sum + epsilon
    

    dice_arr = 2. * and_mask_sum_arr / delimiter_arr

    return dice_arr.tolist()