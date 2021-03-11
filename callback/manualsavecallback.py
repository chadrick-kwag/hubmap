import os, torch

class ManualSaveCallback:

    def __init__(self, savedir, net) -> None:

        assert os.path.exists(savedir)

        self.savedir = savedir
        self.net = net

    def run(self, epoch, step):

        savepath = os.path.join(self.savedir, f'epoch={epoch}_step={step}.pt')

        torch.save(self.net.state_dict(), savepath)
        