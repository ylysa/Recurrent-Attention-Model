import torch
import utils
import generate_dataset
from trainer import Trainer
from config import get_config

def main(config):
    utils.prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {"num_workers": 1, "pin_memory": True}

    completedataset = generate_dataset.generateDataset()
    dataset = completedataset[:300]
    dataset = dataset * 170

    valdataset = completedataset[300:600]
    testdataset = completedataset[:600]

    images = generate_dataset.extractImg(dataset)
    valimages = generate_dataset.extractImg(valdataset)
    testimages = generate_dataset.extractImg(testdataset)

    trainer = Trainer(config, dataset, valdataset, testdataset, images, valimages, testimages)

    # either train
    if config.is_train:
        utils.save_config(config)
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
