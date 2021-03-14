#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from numpy import argmax

def plot_stats(stats, stats_label, title, filename):
    train, val = stats
    train_label, val_label = stats_label

    fig = plt.figure()
    plt.plot(train, label=train_label)
    plt.plot(val, label=val_label)
    plt.xlabel(train_label.split(' ')[-1])
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()
    fig.savefig(filename)

def record_stats(loss, accuracy, lr):
    train_loss, val_loss = loss
    training_acc, val_acc = accuracy
    with open('crossentropyloss.csv', 'a') as f:
        f.write(f'\nLearning rate {lr}\n')
        f.write('Epochs,Train loss,Val loss,Train accuracy,Validation accuracy\n')
        for i in range(len(train_loss)):
            f.write(f'{i},{train_loss[i]},{val_loss[i],training_acc[i],val_acc[i]}\n') 
    
    plot_stats(stats=(train_loss, val_loss), 
               stats_label=('Training loss', 'Validation loss'), 
               title=f'Loss over epochs, learning rate {lr}',
               filename='/figure/loss.png')
    plot_stats(stats=(training_acc, val_acc), 
               stats_label=('Training accuracy', 'Validation accuracy'), 
               title=f'Accuracy over epochs, learning rate {lr}',
               filename='/figure/accuracy.png')

def correct_classified(pred, truth):
    return accuracy_score(truth.cpu().detach().numpy(), argmax(pred.cpu().detach().numpy(), axis=1), normalize=False)

if __name__ == "__main__":
    record_stats(loss=([1, 2, 3, 10], [0, -1, 3, 5]), 
                 accuracy=([0, -10, 2, 0.5], [0, 10, 20, 50]), 
                 lr=1e-5)

