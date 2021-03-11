from pathlib import Path
import pickle

import tools.file_io as file_io
from tools.argument_parsing import get_argument_parser
from data_handler.dataset import ASDataset, get_data_loader
import logging
from modules.triresnet import TridentResNet
import torch
import torch.nn as nn
import numpy as np

def dataset(settings_data, hpram_settings, is_testing):
    batch_size = hpram_settings["batch_size"]
    num_workers = hpram_settings["num_workers"]
    if not is_testing:
        train_set = ASDataset(
            split="train",
            load_into_memory=False,
            data_features_dir=settings_data["data_features_dir"],
            data_parent_dir=settings_data["data_parent_dir"],
            meta_parent_dir=settings_data["meta_parent_dir"],
            meta_dir=settings_data["meta_train_dir"],
        ) 
        train_loader = get_data_loader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers = num_workers,
        )

        val_set = ASDataset(
            split="val",
            load_into_memory=False,
            data_features_dir=settings_data["data_features_dir"],
            data_parent_dir=settings_data["data_parent_dir"],
            meta_parent_dir=settings_data["meta_parent_dir"],
            meta_dir=settings_data["meta_val_dir"],
        ) 

        val_loader = get_data_loader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers = num_workers,
        )

        eva_set = ASDataset(
            split="eva",
            load_into_memory=False,
            data_features_dir=settings_data["data_features_dir"],
            data_parent_dir=settings_data["data_parent_dir"],
            meta_parent_dir=settings_data["meta_parent_dir"],
            meta_dir=settings_data["meta_eva_dir"],
        ) 

        eva_loader = get_data_loader(
            eva_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers = num_workers,
        )
        return train_loader, val_loader, eva_loader

    else:
        test_set = ASDataset(
            split="test",
            load_into_memory=False,
            data_features_dir=settings_data["data_eval_features_dir"],
            data_parent_dir=settings_data["data_parent_dir"],
            meta_parent_dir=None,
            meta_dir=None,
        ) 
        test_loader = get_data_loader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers = num_workers,
        )
        return test_set


def _do_training(model, optimizer, loss_fn, scheduler, train_loader, val_loader, epochs, device):
    train_loss = []
    val_loss = []
    val_accs = []
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train() 
        for feature,target in train_loader:
            feature = feature.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            y_hat = model(feature)

            loss = loss_fn(y_hat, target.squeeze(1))

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            for feature,target in val_loader:
                feature = feature.to(device)
                target = target.to(device)
                y_hat = model(feature)

                loss = loss_fn(y_hat, target.squeeze(1))
                val_loss.append(loss.item())
                val_acc = (y_hat.argmax(axis=1) == target.squeeze(1)).float().mean()
                val_accs.append(val_acc.item())
                if loss < best_loss:
                    best_loss = loss.item()
                    torch.save(model.state_dict(), f"triresnet_1.pt")


        scheduler.step()
        log.info(f"Epoch {1}: Train loss: {np.mean(train_loss)} | Val loss: {np.mean(val_loss)} | Val acc: {np.mean(val_accs)}")

def main():
    args = get_argument_parser().parse_args()

    file_dir = args.file_dir
    config_file = args.config_file
    file_ext = args.file_ext
    verbose = args.verbose
    job_id = args.job_id

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    log = logging.getLogger(__name__)
    settings = file_io.load_yaml_file(Path(
        file_dir, f'{config_file}.{file_ext}'))

    dataset_settings = settings["dataset"]
    model_settings = settings["model"]
    hpram_settings = settings["hyperparams"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading model")
    model = TridentResNet(pretrained=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.3, verbose=True)
    loss_fn = nn.CrossEntropyLoss()

    if settings["flow"]["training"]:
        log.info("Loading training and validation data")
        train_loader, val_loader, eva_loader = dataset(dataset_settings, hpram_settings, is_testing=False)
        log.info(f'Training data shape: {len(train_loader)}')
        log.info(f'Validation data shape: {len(val_loader)}')
        log.info(f'Evaluation data shape: {len(eva_loader)}')

        _do_training(model,optimizer, loss_fn, scheduler, train_loader, 
                    val_loader, hpram_settings["epochs"], device)
    if settings["flow"]["testing"]:
        testing_loader = dataset(dataset_settings, hpram_settings, is_testing=True)
if __name__ == '__main__':
    main()
