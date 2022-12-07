from dataloader import SCANDatasetLoader
# from utils import DATA_CLOUD

DATA_CLOUD = "SCAN/SCAN-master"

def load_data(train=True):
    if train:
        data = SCANDatasetLoader(root=DATA_CLOUD, split_type="simple_split")
    else:
        data = SCANDatasetLoader(root=DATA_CLOUD, split_type="simple_split", train=False)

    return data
