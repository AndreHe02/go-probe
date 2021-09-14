from go_probe.experiments import DefaultExperiment
from go_probe.datasets import CrossValDataset
from go_probe.models import load_control_model
import numpy as np

class CtrlExperiment(DefaultExperiment):

    channels = [8, 64, 64, 64, 48, 48, 32, 32]
    probe_dims = [nc *19*19 for nc in channels]
    label_dim = 90
    num_epochs = 5

    def __init__(self, dataset):
        super(CtrlExperiment, self).__init__(dataset)
        model = load_control_model()
        self.model = model.cuda()

    def dataloader(self, split):
        if split == 'train':
            self.dataset.shuffle(split)
        return self.dataset.loader(split, num_workers=16, max_ram_files=40)

    def get_internal_reps(self, X):
        reps = self.model.forward_layer_outputs(X)
        return [rep.flatten(start_dim=1).cuda() for rep in reps]

class CtrlPatternExperiment(CtrlExperiment):
    label_dim = 4
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir')
    parser.add_argument('-n', '--n_fold', default=10)
    args = parser.parse_args()
    data_dir = args.data_dir
    n_fold = int(args.n_fold)

    print('Probing natural language features')
    bow_cvd = CrossValDataset(data_dir, 'svp', 'bow', n_fold, 512)
    bow_metrics = []
    for i in range(n_fold):
        dataset = bow_cvd.val_split(i)
        exp = CtrlExperiment(dataset)
        fold_metrics = exp.run()
        bow_metrics.append(fold_metrics)
    np.save('control_bow_aucs.npy', np.stack(bow_metrics))

    print('Probing pattern based features')
    pattern_cvd = CrossValDataset(data_dir, 'svp', 'patterns', n_fold, 512)
    pattern_metrics = []
    for i in range(n_fold):
        dataset = pattern_cvd.val_split(i)
        exp = CtrlPatternExperiment(dataset)
        fold_metrics = exp.run()
        pattern_metrics.append(fold_metrics)
    np.save('control_pattern_aucs.npy', np.stack(pattern_metrics))
