import os, torch


class MonitorCkptSaveCallback:

    def __init__(self, net, savedir, mode, metric_name, keep_size = 3) -> None:

        self.net = net
        self.savedir = savedir
        self.metric_name = metric_name

        assert mode in ['max', 'min'], f'invalid mode={mode}'
        self.mode = mode

        
        self.keep_size = keep_size

        self.save_list = []

    def run(self, epoch, step, global_step, val):

        savepath = os.path.join(self.savedir, f'{self.metric_name}_{val}_epoch={epoch}_step={step}.pt')

        new_item = (val, savepath)

        new_save_list = self.save_list + [new_item]

        if self.mode == 'max':
            reverse_flag = True
        elif self.mode == 'min':
            reverse_flag = False
        else:
            raise Exception(f'invalid mode: {self.mode}')

        new_save_list.sort(key=lambda x: x[0], reverse=reverse_flag)

        if len(new_save_list) > self.keep_size:
            eliminate_item = new_save_list[self.keep_size]
        
            os.remove(eliminate_item[1])
        

        torch.save(self.net.state_dict(), savepath)



    
