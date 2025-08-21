from torch.utils.data import Dataset


class LAMConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumsum_lengths = [0] + [sum(self.lengths[: i + 1]) for i in range(len(self.lengths))]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        dataset_idx = 0
        while dataset_idx < len(self.cumsum_lengths) - 1 and idx >= self.cumsum_lengths[dataset_idx + 1]:
            dataset_idx += 1
        local_idx = idx - self.cumsum_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]
