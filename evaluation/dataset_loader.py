from datasets import load_dataset


class DatasetLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def dataset_load(self):
        dataset = load_dataset("ai-forever/LIBRA", self.dataset_name)["test"]
        return dataset
