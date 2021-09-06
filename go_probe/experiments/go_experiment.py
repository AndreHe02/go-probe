from go_probe.datasets.datasets import DefaultDataset
from go_probe.experiments import DefaultExperiment
from go_probe.datasets import CrossValDataset, pattern_features
from go_probe.models import load_go_model
import numpy as np

class GoExperiment(DefaultExperiment):

    channels = [8, 64, 64, 64, 48, 48, 32, 32]
    probe_dims = [nc *19*19 for nc in channels]
    label_dim = 90
    num_epochs = 5

    def __init__(self, dataset, model_weights):
        super(GoExperiment, self).__init__(dataset)
        model = load_go_model(model_weights)
        self.model = model.cuda()

    def dataloader(self, split):
        if split == 'train':
            self.dataset.shuffle(split)
        return self.dataset.loader(split, num_workers=4, max_ram_files=20)

    def get_internal_reps(self, X):
        reps = self.model.forward_layer_outputs(X)
        return [rep.flatten(start_dim=1).cuda() for rep in reps]

class GoPatternExperiment(GoExperiment):
    label_dim = 4
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir')
    parser.add_argument('-n', '--n_fold', default=10)
    parser.add_argument('-w', '--weights', default='go_model_weights.tar')
    args = parser.parse_args()
    data_dir = args.data_dir
    n_fold = int(args.n_fold)
    weights_dir = args.weights

    print('Probing natural language features')
    bow_cvd = CrossValDataset(data_dir, 'svp', 'bow', n_fold, 512)
    bow_metrics = []
    for i in range(n_fold):
        dataset = bow_cvd.val_split(i)
        exp = GoExperiment(dataset, weights_dir)
        fold_metrics = exp.run()
        bow_metrics.append(fold_metrics)
    np.save('imitation_bow_aucs.npy', np.stack(bow_metrics))

    print('Probing pattern based features')
    pattern_cvd = CrossValDataset(data_dir, 'svp', 'patterns', n_fold, 512)
    pattern_metrics = []
    for i in range(n_fold):
        dataset = pattern_cvd.val_split(i)
        exp = GoPatternExperiment(dataset, weights_dir)
        fold_metrics = exp.run()
        pattern_metrics.append(fold_metrics)
    np.save('imitation_pattern_aucs.npy', np.stack(pattern_metrics))