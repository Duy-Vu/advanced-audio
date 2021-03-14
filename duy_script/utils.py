#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from numpy import argmax

def plot_stats(ax, stats, stats_label, title):
    train, val = stats
    train_label, val_label = stats_label

    ax.plot(train, label=train_label)
    ax.plot(val, label=val_label)
    ax.set_xlabel(train_label.split(' ')[-1])
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()

def record_stats(loss, accuracy, hyperparam):
    train_loss, val_loss = loss
    training_acc, val_acc = accuracy
    with open('crossentropyloss.csv', 'a') as f:
        f.write(f'\nHyperameters: {hyperparam}\n')
        f.write('Epochs,Train loss,Val loss,Train accuracy,Validation accuracy\n')
        for i in range(len(train_loss)):
            f.write(f'{i},{train_loss[i]},{val_loss[i],training_acc[i],val_acc[i]}\n') 

    # Plot figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), tight_layout=True)
    fig.suptitle(f'{hyperparam}', fontsize=12)

    plot_stats(axs[0],
               stats=(train_loss, val_loss), 
               stats_label=('Training loss', 'Validation loss'), 
               title=f'Loss over epochs')
    plot_stats(axs[1],
               stats=(training_acc, val_acc), 
               stats_label=('Training accuracy', 'Validation accuracy'), 
               title=f'Accuracy over epochs')
    plt.show()
    fig.savefig(f'figure/model_{hyperparam}.png')

def correct_classified(pred, truth):
    return accuracy_score(truth.cpu().detach().numpy(), argmax(pred.cpu().detach().numpy(), axis=1), normalize=False)

if __name__ == "__main__":
    record_stats(loss=([1, 2, 3, 10], [0, -1, 3, 5]), 
                 accuracy=([0, -10, 2, 0.5], [0, 10, 20, 50]), 
                 hyperparam={'lr': 1e-5, 'weight decay':1e-5})

