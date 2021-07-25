from go_probe.experiments import DefaultExperiment
from go_probe.datasets import CrossValDataset, DefaultDataset
from go_probe.models import load_elf_model
from go_probe.utils.io import write_pkl

class ELFExperiment(DefaultExperiment):

    channels = [256] * 20
    probe_dims = [nc *19*19 for nc in channels]
    label_dim = 60
    num_epochs = 5

    def __init__(self, dataset, model_weights):
        super(ELFExperiment, self).__init__(dataset)
        model = load_elf_model(model_weights)
        self.model = model.cuda()

    def dataloader(self, split):
        if split == 'train':
            self.dataset.shuffle(split)
        return self.dataset.loader(split, num_workers=2, max_ram_files=10)

    def get_internal_reps(self, X):
        for rep in self.model.resnet_layer_output_generator(X):
            yield rep.flatten(start_dim=1).cuda()

if __name__ == "__main__":
    dataset = DefaultDataset('labels', 'C:/Users/andre/Documents/data/go/annotated/elf_bow', 0.9, 0.1, 32)
    exp = ELFExperiment(dataset, 'C:/Users/andre/Desktop/go-probe/elf_model_weights.bin')
    metrics = exp.run()
    write_pkl(metrics, "word_probe_metrics_ELF.pkl")