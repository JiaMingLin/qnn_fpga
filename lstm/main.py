import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

from data.EMG.dataset import *
from lstm import SeqModel
from opts import parser
from tqdm import tqdm

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
    hidden_size = args.hidden
    num_layers = args.num_layers
    quant = args.quant
    w_bit = args.w_bit
    a_bit = args.a_bit
    i_bit = args.i_bit
    r_bit = args.r_bit
    no_brevitas = args.no_brevitas

    '''
    STEP 1: LOADING DATASET
    '''
    if dataset == 'mnist':
        input_size = 28
        seq_size = 28 
        output_size = 10
        train_dataset = dsets.MNIST(root='./data', 
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    download=True)
 
        test_dataset = dsets.MNIST(root='./data', 
                                   train=False, 
                                   transform=transforms.ToTensor())
    elif dataset == 'emg':
        input_size = 8
        seq_size = 100
        output_size = 8
        train_dataset = EGMDataset(path = './data/EMG')
        test_dataset = EGMDataset(path = './data/EMG', train=False)
 

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=True,
                                                num_workers=12)
 
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False,
                                              num_workers=12)

    '''
    STEP 4: INSTANTIATE MODEL CLASS
    '''
    model = SeqModel(input_size, hidden_size, output_size = output_size,
                      num_layers = num_layers,
                      quant = quant, w_bit=w_bit, a_bit=a_bit, i_bit=i_bit, r_bit=r_bit,
                      no_brevitas = no_brevitas)

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
    learning_rate = 0.00025
 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    '''
    STEP 7: TRAIN THE MODEL
    '''
    for epoch in range(num_epochs):
        train_loss = 0
        train_samples = 0
        for batch_samples, labels in tqdm(train_loader):
            
            batch_samples = batch_samples.cuda()
            labels = labels.cuda()
          
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
         
            # Forward pass to get output/logits
            # outputs.size() --> 100, 10
            outputs = model(batch_samples)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            # print(labels)

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
                        batch_samples = batch_samples.view(-1, seq_size, input_size)
                    batch_samples = batch_samples.cuda()
                    labels = labels.cuda()
                else:
                    batch_samples = batch_samples.view(-1 , seq_size, input_size)
                
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