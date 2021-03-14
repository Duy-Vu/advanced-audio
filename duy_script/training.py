#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
from copy import deepcopy
from resnet_d import ResNet_d
from data_handling import MyDataset, getting_dataset, getting_data_loader

import torch
from torch.optim import Adam
from torch import no_grad, cuda, save, load
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from utils import record_stats, correct_classified


# Check if CUDA is available, else use CPU
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Process on {device}', end='\n\n')

def main():
    # Initialize hyperparameters for the model
    params = {
        'epochs' : 200,
        'batch_size' : 32,
        'learning_rate' : 6e-4,
        'weight_decay' : 1e-4,    
    }
    print(f'{params}')
    lowest_validation_loss = 2.3
    highest_validation_accuracy = 0.1
    best_validation_epoch = 0
    patience = 30
    patience_counter = 0

    # Initialize model, optimizer and loss function 
    resnet_model = ResNet_d().to(device)
    loss_function = CrossEntropyLoss()
    optimizer = Adam(params=resnet_model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay']) 
    #scheduler = MultiStepLR(optimizer, milestones=[50, 85, 120, 145], gamma=0.1, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.3, verbose=True)

    # Load training, validation, and testing dataset.
    print("Getting dataset\n")
    data_dir = "log_mel"
    training_data, validation_data = getting_dataset(data_dir)
    training_loader, num_train = getting_data_loader("training", training_data, params['batch_size'])
    validation_loader, num_valid = getting_data_loader("validation", validation_data, params['batch_size'])
    testing_loader, num_test = getting_data_loader("testing", MyDataset(data_dir='testing'), params['batch_size'])

    # Initialize variables to keep track of performance over epochs
    loss_training, loss_validation, acc_training, acc_validation =  [], [], [], [] 

    # Start training.
    best_model = None
    for epoch in range(1, params['epochs']+1):
        # Counting time each epoch
        start_time = time.time() 

        # Indicate that we are in training mode
        print("\nStart training.")
        resnet_model.train()
        epoch_loss_training, epoch_acc_training = [], 0
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
        epoch_loss_validation, epoch_acc_validation = [], 0
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
        epoch_loss_training = np.array(epoch_loss_training).mean()
        epoch_loss_validation = np.array(epoch_loss_validation).mean()
        epoch_acc_training /= num_train
        epoch_acc_validation /= num_valid
        
        try:        # If MultiStepLR is used
            scheduler.step()
        except:     # If ReduceLROnPlateau is used
            scheduler.step(epoch_acc_validation)
        
        # Check early stopping conditions.
        if epoch_acc_validation > highest_validation_accuracy:
            patience_counter = 0
            lowest_validation_loss = epoch_loss_validation
            highest_validation_accuracy = epoch_acc_validation
            best_validation_epoch = epoch
            # Copy best model
            best_model = deepcopy(resnet_model.state_dict()) 
            print("best model found")
        else:
            patience_counter += 1

        # If the model cannot improve anymore or if we are running with enough epochs, do the testing.
        if patience_counter >= patience or epoch == params['epochs']:
            if patience_counter >= patience:
                print('\nExiting due to early stopping', end='\n\n')
            else:
                print('\nDone training!', end='\n\n')

            print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss} ' 
                  f'and accuracy {highest_validation_accuracy}', end='\n\n')
            if best_model is None:
                print('No best model. ')
            else:
                print('Saving the best model\n')
                save({'model_state_dict': best_model,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': lowest_validation_loss,
                    'accuracy': highest_validation_accuracy,
                    }, f'save_models/model{params}.th') 

                # Test with the best model peformed on validation set
                testing(resnet_model, f'save_models/model{params}.th', testing_loader, num_test)
                break
        
        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss: {epoch_loss_training:7.4f} | '
              f'Mean validation loss: {epoch_loss_validation:7.4f} | '
              f'Training accuracy: {epoch_acc_training:7.5f} | '
              f'Validation accuracy: {epoch_acc_validation:7.5f} | '
              f'Time: {time.time() - start_time}')
        
    loss_training.append(epoch_loss_training)
    loss_validation.append(epoch_loss_validation) 
    acc_training.append(epoch_acc_training)    
    acc_validation.append(epoch_acc_validation)

    record_stats(loss=(loss_training, loss_validation), 
                 accuracy=(loss_validation, acc_validation), 
                 hyperparam=params)

def testing(suboptimal_model, model_path, test_loader, test_total):
    print('Start testing', end=' | ')
    testing_loss = []
    testing_acc = 0
    loss_fn = CrossEntropyLoss()
    
    with no_grad():
        for batch in test_loader:
            x_test, y_test = batch
            y_test = torch.LongTensor(y_test)  
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            y_hat = suboptimal_model(x_test)
            loss = loss_fn(y_hat, y_test)
            testing_loss.append(loss.item())
            testing_acc += correct_classified(y_hat, y_test)

    testing_loss = np.array(testing_loss).mean()
    testing_acc /= test_total
    print("Suboptimal")
    print(f'Testing loss: {testing_loss:7.4f} | Testing accuracy: {testing_acc:7.5f}')

    print("Optimal")
    testing_loss = []
    testing_acc = 0
    loss_fn = CrossEntropyLoss()

    checkpoint = load('save_models/' + model_path, map_location=device)
    test_model = ResNet_d()
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.to(device)
    with no_grad():
        for batch in test_loader:
            x_test, y_test = batch
            y_test = torch.LongTensor(y_test)  
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            y_hat = test_model(x_test)
            loss = loss_fn(y_hat, y_test)
            testing_loss.append(loss.item())
            testing_acc += correct_classified(y_hat, y_test)

    testing_loss = np.array(testing_loss).mean()
    testing_acc /= test_total
    print(f'Testing loss: {testing_loss:7.4f} | Testing accuracy: {testing_acc:7.5f}')

if __name__ == '__main__':
    main()