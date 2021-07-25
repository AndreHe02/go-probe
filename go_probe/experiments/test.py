from go_probe.datasets.datasets import DefaultDataset
from go_probe.experiments import DefaultExperiment
from go_probe.datasets import CrossValDataset
from go_probe.models import load_go_model
from go_probe.utils.io import write_pkl
import torch.nn as nn
import torch.optim as optim

class TestExperiment(DefaultExperiment):

    channels = [8, 64, 64, 64, 48, 48, 32, 32]
    probe_dims = [nc *19*19 for nc in channels]
    label_dim = 60
    num_epochs = 5

    def __init__(self, dataset, model_weights):
        super(TestExperiment, self).__init__(dataset)
        model = load_go_model(model_weights)
        self.model = model.cuda()

    def dataloader(self, split):
        if split == 'train':
            self.dataset.shuffle(split)
        return self.dataset.loader(split, max_ram_files = 20)#num_workers=4, max_ram_files=20)

    def get_internal_reps(self, X):
        reps = self.model.forward_layer_outputs(X)
        return [rep.flatten(start_dim=1).cuda() for rep in reps]

if __name__ == "__main__":
    dataset = CrossValDataset('labels', 'C:/Users/andre/Documents/data/go/annotated/go_bow', 10, 1024).val_split(0)
    exp = TestExperiment(dataset, 'C:/Users/andre/Desktop/go-probe/go_model_weights.tar')
    metrics = exp.run()
    print(metrics)
