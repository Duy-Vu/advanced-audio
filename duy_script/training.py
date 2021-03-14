#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

import torch
from torch import no_grad, cuda, save, load
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from resnet_d import ResNet_d
from data_handling import MyDataset, getting_dataset, getting_data_loader

import time

from utils import record_stats, correct_classified


def main():
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Initialize hyperparameters for the model
    epochs = 250
    batch_size = 32
    learning_rate = 2e-4
    weight_decay = 0    # weight_decay=1e-4

    lowest_validation_loss = 2.3
    highest_validation_accuracy = 0.1
    best_validation_epoch = 0
    patience = 30
    patience_counter = 0

    # Initialize model, optimizer and loss function 
    resnet_model = ResNet_d()
    resnet_model = resnet_model.to(device)
    optimizer = Adam(params=resnet_model.parameters(), lr=learning_rate, weight_decay=weight_decay) 
    loss_function = CrossEntropyLoss()

    # Load training, validation, and testing dataset.
    print("Getting dataset\n")
    data_dir = "log_mel"
    training_data, validation_data = getting_dataset(data_dir)
    training_loader, num_train = getting_data_loader("training", training_data, batch_size)
    validation_loader, num_valid = getting_data_loader("validation", validation_data, batch_size)
    testing_loader, num_test = getting_data_loader("testing", MyDataset(data_dir='testing'), batch_size)

    # Start training.
    best_model = None
    for epoch in range(epochs):
        # Counting time per epoch
        start_time = time.time() 

        epoch_loss_training, epoch_loss_validation = [], []
        epoch_acc_training, epoch_acc_validation = 0, 0

        # Indicate that we are in training mode
        print("\nStart training.")
        resnet_model.train()
        for batch in training_loader:
            # Zero the gradient of the optimizer.
            optimizer.zero_grad()
            
            x_train, y_train = batch
            y_train = torch.LongTensor(y_train)   
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            
            # Get the predictions of our model.
            y_hat = resnet_model(x_train)

            # Calculate the loss of our model.
            loss = loss_function(input=y_hat, target=y_train) 

            # Find the number of accurate predictions of our model
            epoch_acc_training += correct_classified(y_hat, y_train)

            # Backpropagation
            loss.backward()
            optimizer.step()
            epoch_loss_training.append(loss.item())

        # Indicate that we are in evaluation mode
        print("Start validating.")
        resnet_model.eval()
        with no_grad():
            for batch in validation_loader:
                x_val, y_val = batch
                y_val = torch.LongTensor(y_val)  
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                # Get the predictions of the model
                y_hat = resnet_model(x_val)

                # Calculate the loss and accuracy
                loss = loss_function(y_hat, y_val)
                epoch_loss_validation.append(loss.item())
                epoch_acc_validation += correct_classified(y_hat, y_val)
        
        # Calculate final mean losses and accuracy
        epoch_loss_validation = np.array(epoch_loss_validation).mean()
        epoch_loss_training = np.array(epoch_loss_training).mean()
        epoch_acc_training /= num_train
        epoch_acc_validation /= num_valid

        # Check early stopping conditions.
        if epoch_loss_validation < lowest_validation_loss:
            patience_counter = 0
            lowest_validation_loss = epoch_loss_validation
            highest_validation_accuracy = epoch_acc_validation
            best_validation_epoch = epoch
            # Copy best model
            best_model = deepcopy(resnet_model.state_dict()) 
        else:
            patience_counter += 1

        # If the model cannot improve anymore or if we are running with enough epochs, do the testing.
        if patience_counter >= patience or epoch == epochs-1:
            print('\nExiting due to early stopping', end='\n\n')
            print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss} ' 
                  f'and accuracy {highest_validation_accuracy}', end='\n\n')
            if best_model is None:
                print('No best model. ')
            else:
                print('Saving the best model\n')
                save({'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': lowest_validation_loss,
                    'accuracy': highest_validation_accuracy,
                    }, f'/save_models/good_model.th')

                # Test with the best model peformed on validation set
                print('Start testing', end=' | ')
                testing_loss = []
                testing_acc = 0
                best_model.eval()
                with no_grad():
                    for batch in testing_loader:
                        x_test, y_test = batch
                        y_test = torch.LongTensor(y_test)  
                        x_test = x_test.to(device)
                        y_test = y_test.to(device)

                        y_hat = resnet_model(x_test)
                        loss = loss_function(y_hat, y_test)
                        testing_loss.append(loss.item())
                        testing_acc += correct_classified(y_hat, y_test)

                testing_loss = np.array(testing_loss).mean()
                testing_acc /= num_test
                print(f'Testing loss: {testing_loss:7.4f} | Testing accuracy: {testing_acc:7.5f}')
                break

        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss: {epoch_loss_training:7.4f} | '
              f'Mean validation loss: {epoch_loss_validation:7.4f} | '
              f'Training accuracy: {epoch_acc_training:7.5f} | '
              f'Validation accuracy: {epoch_acc_validation:7.5f} | '
              f'Time: {time.time() - start_time}')

    record_stats(loss=(epoch_loss_training.tolist(), epoch_loss_validation.tolist()), 
                 accuracy=(epoch_acc_training.tolist(), epoch_acc_validation.tolist()), 
                 lr=learning_rate)

if __name__ == '__main__':
    main()
    """
    file = ''
    checkpoint = load('save_models/' + file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']

    model.eval()
    # - or -
    model.train()

    """