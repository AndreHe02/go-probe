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

    def forward_layer_outputs(self, x):
        layer_outputs = [x]
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.nonlinear(x)
            layer_outputs.append(x.detach())
        return layer_outputs

def load_go_model(path, rm_prefix=True):
    model = GoModel(None)
    checkpoint = torch.load(path, map_location=torch.device('cuda'))
    state_dict = checkpoint['state_dict']
    if rm_prefix:
        state_dict = {key[7:]:state_dict[key] for key in state_dict}
    model.load_state_dict(state_dict)
    return model

def load_control_model():
    model = GoModel(None)
    return model