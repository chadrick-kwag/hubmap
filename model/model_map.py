from .v2 import  EfficientUnetb0, EfficientUnetb1, EfficientUnetb2
from .v1 import model_v1

model_map = {
    'model_v1': model_v1,
    # 'EfficientUnetb7': EfficientUnetb7,
    'EfficientUnetb0': EfficientUnetb0,
    'EfficientUnetb1': EfficientUnetb1,
    'EfficientUnetb2': EfficientUnetb2
}