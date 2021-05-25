from dlgo.agent.predict import DeepLearningAgent
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from go_model import GoModel
import torch
import numpy as np
from dlgo.gtp import GTPFrontend

class Dummy:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        X = np.concatenate((X, np.ones((1, 1, 19, 19))), axis=1)
        y = self.model(torch.Tensor(X)).detach()
        return torch.nn.functional.softmax(y, dim=1).numpy()

go_model = GoModel(None)
checkpoint = torch.load('C:/Users/andre/Desktop/go-probe/model_ckpt.pth.tar', map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict']
state_dict = {key[7:]:state_dict[key] for key in state_dict} #remove 'module.' prefix
go_model.load_state_dict(state_dict)

encoder = SevenPlaneEncoder((19, 19))
dlagent = DeepLearningAgent(Dummy(go_model), encoder)

frontend = GTPFrontend(dlagent)
frontend.run()

#random_agent = RandomBot()
#web_app = get_web_app({'predict': dlbot})
#web_app = get_web_app({'random': random_agent})
#web_app.run()
