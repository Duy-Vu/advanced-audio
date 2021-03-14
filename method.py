from pathlib import Path
import pickle

import tools.file_io as file_io
from tools.argument_parsing import get_argument_parser
from data_handler.dataset import ASDataset, get_data_loader, split_dataset
import logging
from modules.triresnet import TridentResNet
from modules.triresnet_2 import TridentResNet2
from modules.resnet_d import ResNet_d
from modules.focal_loss import FocalLoss
import torch
import torch.nn as nn
import numpy as np
import time
import sklearn
import matplotlib.pyplot as plt

args = get_argument_parser().parse_args()

logging.basicConfig(
    filename=f"logs/log_{args.job_id}",
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

def plot_util(epochs, train, val, title):
    epochs = range(epochs)
    plt.plot(epochs, loss_train, 'g', label=f'Training {title}')
    plt.plot(epochs, loss_val, 'b', label=f'Validation {title}')
    plt.title(f'Training and Validation {title}')
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(f"{args.job_id}_{title}.png")

def dataset(settings_data, hpram_settings, is_testing):
    batch_size = hpram_settings["batch_size"]
    num_workers = hpram_settings["num_workers"]
    if not is_testing:
        train_set, val_set = split_dataset(
            ASDataset(
                split="train",
                load_into_memory=False,
                data_features_dir=settings_data["data_features_dir"],
                data_parent_dir=settings_data["data_parent_dir"],
                meta_parent_dir=settings_data["meta_parent_dir"],
                meta_dir=settings_data["meta_train_dir"],
            ) 
        )
        train_loader = get_data_loader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers = num_workers,
        )

        """
        val_set = ASDataset(
            split="val",
            load_into_memory=False,
            data_features_dir=settings_data["data_features_dir"],
            data_parent_dir=settings_data["data_parent_dir"],
            meta_parent_dir=settings_data["meta_parent_dir"],
            meta_dir=settings_data["meta_val_dir"],
        ) 
        """
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


def _do_training(model, optimizer, loss_fn, scheduler, train_loader, val_loader, settings, device):
    best_acc = 0
    best_epoch = 0
    patience = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(settings["epochs"]):
        train_loss = []
        val_loss = []
        val_accs = []
        start_time = time.time()
        model.train() 
        pred_train = []
        targets_train = []
        for feature,target in train_loader:
            feature = feature.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            y_hat = model(feature)

            loss = loss_fn(y_hat, target.squeeze(1))

            pred_train.extend(y_hat.argmax(axis=1).cpu().numpy().tolist())
            targets_train.extend(target.squeeze(1).cpu().numpy().tolist())

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_acc = sklearn.metrics.accuracy_score(pred_train,targets_train)
        model.eval()
        with torch.no_grad():
            pred = []
            targets = []
            val_acc = 0
            for feature,target in val_loader:
                feature = feature.to(device)
                target = target.to(device)
                y_hat = model(feature)

                loss = loss_fn(y_hat, target.squeeze(1))
                val_loss.append(loss.item())
                pred.extend(y_hat.argmax(axis=1).cpu().numpy().tolist())
                targets.extend(target.squeeze(1).cpu().numpy().tolist())
            val_acc = sklearn.metrics.accuracy_score(pred,targets)
            scheduler.step(np.mean(val_loss))
        
        # Append to the big lists
        train_losses.append(np.mean(train_loss))
        val_losses.append(np.mean(val_loss))
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping
        if val_acc > best_acc:
            patience = 0
            best_loss = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), f'save_models/{settings["save_model"]}')
        else:
            patience += 1
        if patience > settings['patience']:
            log.info(f'Early stopping, best loss at {best_epoch}')
            plot_util(epoch, train_losses, val_losses, "Loss")
            plot_util(epoch, train_accs, val_s, "Loss")
            return
        else:
            log.info(f"Epoch {epoch}: Train loss: {np.mean(train_loss)} | Train acc: {train_acc} |Val loss: {np.mean(val_loss)} | Val acc: {val_acc} | Time: {time.time()-start_time}")

def _do_evaluation(model,eva_loader, settings, device):
    checkpoint = torch.load(f'save_models/{settings["save_model"]}', map_location=device)
    model.load_state_dict(checkpoint)

    pred = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch in eva_loader:
            if len(batch)>1:
                feature, target = batch
                target = target.to(device)
            else:
                feature = batch
            feature = feature.to(device)
            y_hat = model(feature)

            pred.extend(y_hat.argmax(axis=1).cpu().numpy().tolist())
            targets.extend(target.squeeze(1).cpu().numpy())
    log.info("Accuracy score")
    log.info(sklearn.metrics.accuracy_score(pred,targets))
    return pred, target 

def main():

    file_dir = args.file_dir
    config_file = args.config_file
    file_ext = args.file_ext
    verbose = args.verbose
    job_id = args.job_id
    
    settings = file_io.load_yaml_file(Path(
        file_dir, f'{config_file}.{file_ext}'))

    dataset_settings = settings["dataset"]
    model_settings = settings["model"]
    hpram_settings = settings["hyperparams"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading model")
    model = None
    if model_settings["model_name"] == "triresnet":
        model = TridentResNet(pretrained=False)
    elif model_settings["model_name"] == "triresnet_2":
        model = TridentResNet2(pretrained=False)
    elif model_settings["model_name"] == "resnet_d":
        model = ResNet_d()
    model.to(device)

    log.info(f"Model saved dir: {hpram_settings['save_model']}")
    if settings["flow"]['continue_training']:
        log.info('Loading model checkpoint')
        checkpoint = torch.load(f'save_models/{hpram_settings["save_model"]}', map_location=device)
        model.load_state_dict(checkpoint)

    optimizer = torch.optim.Adam(model.parameters(), lr=hpram_settings["optimizer"]["lr"])
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.3, verbose=True)
    if hpram_settings['loss'] == "focal":
        loss_fn = FocalLoss(gamma=2)
    else:
        loss_fn = nn.CrossEntropyLoss()

    train_loader, val_loader, eva_loader = dataset(dataset_settings, hpram_settings, is_testing=False)
    log.info(f'Training data shape: {len(train_loader)}')
    log.info(f'Validation data shape: {len(val_loader)}')
    log.info(f'Evaluation data shape: {len(eva_loader)}')
    if settings["flow"]["training"]:
        log.info("Loading training and validation data")
        _do_training(model,optimizer, loss_fn, scheduler, train_loader, 
                    val_loader, hpram_settings, device)
    
    if settings["flow"]["testing"]:
        _do_evaluation(model, eva_loader, hpram_settings, device)
        log.info(f'Predicting on Evaluation set')
        predictions, targets = _do_evaluation(model, eva_loader, hpram_settings, device)

if __name__ == '__main__':
    main()
