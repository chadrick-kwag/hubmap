from .v1 import manual_focal_loss, focal_loss_on_cpu, dice_loss
from functools import partial


loss_map = {
    'focal_loss_gamma_3': partial(focal_loss_on_cpu, gamma=3),
    'dice_loss': dice_loss
}
