import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

from data.EMG.dataset import *
from lstm import *
from opts import parser

def main():
    args = parser.parse_args()
    cuda = True if torch.cuda.is_available() else False
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor    

    torch.manual_seed(125)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(125)

    dataset = args.dataset
    batch_size = args.batch
    num_epochs = args.epoch
    hidden_dim = args.hidden

    '''
    STEP 1: LOADING DATASET
    '''
    if dataset == 'mnist':
        input_dim = 28
        seq_dim = 28 
        output_dim = 10
        train_dataset = dsets.MNIST(root='./data', 
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    download=True)
 
        test_dataset = dsets.MNIST(root='./data', 
                                   train=False, 
                                   transform=transforms.ToTensor())
    elif dataset == 'emg':
        input_dim = 8
        seq_dim = 100
        output_dim = 8
        train_dataset = EGMDataset()
        test_dataset = EGMDataset(train=False)
 

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=True)
 
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False)

    '''
    STEP 4: INSTANTIATE MODEL CLASS
    '''
    layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
    
 
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    #model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)

    #######################
    #  USE GPU FOR MODEL  #
    #######################
 
    if torch.cuda.is_available():
        model.cuda()
     
    '''
    STEP 5: INSTANTIATE LOSS CLASS
    '''
    criterion = nn.CrossEntropyLoss()
 
    '''
    STEP 6: INSTANTIATE OPTIMIZER CLASS
    '''
    learning_rate = 0.001
 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    '''
    STEP 7: TRAIN THE MODEL
    '''
    for epoch in range(num_epochs):
        train_loss = 0
        train_samples = 0
        for i, (batch_samples, labels) in enumerate(train_loader):
            # Load images as Variable
            #######################
            #  USE GPU FOR MODEL  #
            #######################
            if torch.cuda.is_available():
                if len(batch_samples.shape) == 4:
                    batch_samples = batch_samples.view(-1, seq_dim, input_dim)
                batch_samples = batch_samples.cuda()
                labels = labels.cuda()
            else:
              batch_samples = batch_samples.view(-1, seq_dim, input_dim)
              labels = labels
          
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
         
            # Forward pass to get output/logits
            # outputs.size() --> 100, 10
            outputs = model(batch_samples)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            print(labels)

            if torch.cuda.is_available():
                loss.cuda()

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            train_loss += loss.item()
            train_samples += labels.size(0)
        
        train_loss = train_loss / train_samples
        if (epoch+1) % 1 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            val_loss = 0
            for batch_samples, labels in test_loader:
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                if torch.cuda.is_available():
                    if len(batch_samples.shape) == 4:
                        batch_samples = batch_samples.view(-1, seq_dim, input_dim)
                    batch_samples = batch_samples.cuda()
                    labels = labels.cuda()
                else:
                    batch_samples = batch_samples.view(-1 , seq_dim, input_dim)
                
                # Forward pass only to get logits/output
                outputs = model(batch_samples)
                val_loss += criterion(outputs, labels).item()
                
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                 
                # Total number of labels
                total += labels.size(0)
                 
                # Total correct predictions
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()
            val_loss = val_loss / total
            
            accuracy = 100 * correct / total
            # Print Loss
            print('Epoch: {}. Train Loss: {}. Validation Loss: {}, Accuracy: {}'.format(epoch, train_loss, val_loss, accuracy))


if __name__ == "__main__":
    main()