from efficientunet import get_efficientunet_b7, get_efficientunet_b0, get_efficientunet_b1, get_efficientunet_b2
import torch

# b0unet = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=True)


# class EfficientUnetb7(torch.nn.Module):

#     def __init__(self, in_ch):

#         super().__init__()

#         self.base = get_efficientunet_b7(out_channels=in_ch, concat_input=True, pretrained=True)

#     def forward(self, x):
#         y = self.base(x)
#         y = y.permute(0,2,3,1)

#         return y



class EfficientUnetb0(torch.nn.Module):

    def __init__(self, in_ch):

        super().__init__()

        self.base = get_efficientunet_b0(out_channels=in_ch, concat_input=True, pretrained=True)

    def forward(self, x):
        y = self.base(x)
        y = y.permute(0,2,3,1)

        return y


class EfficientUnetb1(torch.nn.Module):

    def __init__(self, in_ch):

        super().__init__()

        self.base = get_efficientunet_b1(out_channels=in_ch, concat_input=True, pretrained=True)

    def forward(self, x):
        y = self.base(x)
        y = y.permute(0,2,3,1)

        return y


class EfficientUnetb2(torch.nn.Module):

    def __init__(self, in_ch):

        super().__init__()

        self.base = get_efficientunet_b2(out_channels=in_ch, concat_input=True, pretrained=True)

    def forward(self, x):
        y = self.base(x)
        y = y.permute(0,2,3,1)

        return y
