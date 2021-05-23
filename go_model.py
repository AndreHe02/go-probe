import torch
import torch.nn as nn 
import torch.nn.functional as F 

class GoModel(nn.Module):
    """
    Following architecture in https://arxiv.org/pdf/1412.3409.pdf
    """
    def __init__(self, args):
        super(GoModel, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(8, 64, 7, padding=3),
                                   nn.Conv2d(64, 64, 5, padding=2),
                                   nn.Conv2d(64, 64, 5, padding=2),
                                   nn.Conv2d(64, 48, 5, padding=2),
                                   nn.Conv2d(48, 48, 5, padding=2),
                                   nn.Conv2d(48, 32, 5, padding=2),
                                   nn.Conv2d(32, 32, 5, padding=2)])
        self.nonlinear = nn.ReLU()
        self.output_linear = nn.Linear(19*19*32, 19*19)


    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.nonlinear(x)
        x = self.output_linear(x.flatten(1))
        return x # batch x NB_CLASSES (361) of scores