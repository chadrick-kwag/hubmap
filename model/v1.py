import torch 




class block_v1(torch.nn.Module):

    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.instnorm1 = torch.nn.InstanceNorm2d(out_ch)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.instnomr2 = torch.nn.InstanceNorm2d(out_ch)
        

    def forward(self, x):

        y = torch.relu(self.instnorm1(self.conv1(x)))

        y = torch.relu(self.instnomr2(self.conv2(y)))

        return y

        
        



class model_v1(torch.nn.Module):

    def __init__(self, in_ch) -> None:
        super().__init__()

        level_out_ch_list = [16,32,64,128,256]
        level_in_ch_list = [in_ch] + level_out_ch_list[:-1]


        self.downblock_list=[]
        self.maxpool2d_list = []

        for in_ch, out_ch in zip(level_in_ch_list, level_out_ch_list):

            b = block_v1(in_ch, out_ch)

            self.downblock_list.append(b)
            self.maxpool2d_list.append(torch.nn.MaxPool2d(2))
        

        high_level_in_ch = level_out_ch_list[-1]
        self.high_level_blocks = []
        

        for _ in range(1):

            self.high_level_blocks.append(block_v1(high_level_in_ch, high_level_in_ch))

        self.upblock_list =[]

        # print(f"high_level_in_ch: {high_level_in_ch}")
        

        skip_in_ch_list = level_out_ch_list[:4]
        skip_in_ch_list.reverse()
        
        rev = level_out_ch_list[:4]
        rev.reverse()

        upsample_in_ch_list = list(level_out_ch_list)
        upsample_in_ch_list.reverse()

        rev = level_out_ch_list[:4]
        rev.reverse()
        up_out_ch_list = rev + [3]
        up_out_ch_list = list(level_in_ch_list)
        up_out_ch_list.reverse()


        # print(f'skip_in_ch_list: {skip_in_ch_list}')
        # print(f'upsample_in_ch_list: {upsample_in_ch_list}')
        # print(f"up_out_ch_list: {up_out_ch_list}")

        self.upsample_list = []

        for upsample_in_ch, up_out_ch in zip(upsample_in_ch_list,up_out_ch_list):
            
            # print(f"upsample_in_ch: {upsample_in_ch}, up_out_ch: {up_out_ch}")
            b = block_v1(upsample_in_ch, up_out_ch)

            self.upblock_list.append(b)
            self.upsample_list.append(torch.nn.Upsample(scale_factor=2))

        # print(f'upblock size: {len(self.upblock_list)}')


        self.last_block = block_v1(3,1)

    def forward(self, x):

        y = x

        down_output_tensor_list = []

        for b, maxpool in zip(self.downblock_list, self.maxpool2d_list):
            y = b(y)
            down_output_tensor_list.append(y)
            y = maxpool(y)
        
        # for t in down_output_tensor_list:
            # print(f'down tensor shp: {t.shape}')

        for b in self.high_level_blocks:
            y = b(y)

        # print(f'high level output: {y.shape}')

        for i in range(len(self.upblock_list)):
            # print(f'upsampling level: {i}')
            upsampled_y = self.upsample_list[i](y)
            skip_tensor = down_output_tensor_list[len(down_output_tensor_list)-1-i]
            # print(f'upsampled_y: {upsampled_y.shape}')
            # print(f"skip_tensor: {skip_tensor.shape}")

            y = self.upblock_list[i](upsampled_y + skip_tensor)

        y = self.last_block(y)
        
        return y







        