from dlgo.agent.predict import DeepLearningAgent
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from go_model import GoModel
import torch
import numpy as np
from dlgo.gtp import GTPFrontend
from dlgo.agent.termination import TerminationAgent, TerminationStrategy
from dlgo import scoring

class GreedyTermination(TerminationStrategy):
    def __init__(self, margin):
        TerminationStrategy.__init__(self)
        self.margin = margin
        self.moved = 0

    def should_pass(self, game_state):
        own_color = game_state.next_player
        if game_state.last_move is None:
            return False
        if game_state.last_move.is_pass:
            game_result = scoring.compute_game_result(game_state)
            if game_result.winner == own_color:
                return True
        return False

    def should_resign(self, game_state):
        own_color = game_state.next_player
        game_result = scoring.compute_game_result(game_state)
        print('WINNING MARGIN' + str(game_result.winning_margin))
        if self.moved:
            if game_result.winner != own_color and game_result.winning_margin >= self.margin:
                return True
        self.moved += 1
        return False

class ModelWrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        X = np.concatenate((X, np.ones((1, 1, 19, 19))), axis=1)
        y = self.model(torch.Tensor(X)).detach()
        return torch.nn.functional.softmax(y, dim=1).numpy()

def main():
    go_model = GoModel(None)
    checkpoint = torch.load('C:/Users/andre/Desktop/go-probe/model_ckpt.pth.tar', map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    state_dict = {key[7:]:state_dict[key] for key in state_dict} #remove 'module.' prefix
    go_model.load_state_dict(state_dict)

    encoder = SevenPlaneEncoder((19, 19))
    dlagent = DeepLearningAgent(ModelWrapper(go_model), encoder)
    agent = TerminationAgent(dlagent, GreedyTermination(40))

    frontend = GTPFrontend(agent)
    frontend.run()

if __name__ == '__main__':
    main()
