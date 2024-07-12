import json


class DatasetLoader():
    def __init__(self, dataset_name, dataset_path=None):
        self.dataset_path = dataset_path
    
    def dataset_load(self):
        dataset = json.load(open(self.dataset_path, encoding="utf-8"))
        return dataset