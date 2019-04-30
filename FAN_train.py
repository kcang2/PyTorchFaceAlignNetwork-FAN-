import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import FAN
from face_dataset import create_dataloader
from face_utils import *


def adjust_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def train(print_every=10):

    max_epoch = 30
    lr = {50: 1e-4, \
        70: 5e-5, \
	90: 1e-5, \
	100: 5e-6}
    batch_size = 32
    
    # Dataloaders for train and test set
    train_dataloader = create_dataloader(root='.', batch_size=batch_size, is_train=True)
    test_dataloader = create_dataloader(root='.', batch_size=32, is_train=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_batches = len(train_dataloader)
    
    # Network configuration
    net = FAN(); net = net.to(device); net = net.train(); net.load_state_dict(torch.load("ckpt_epoch_28"))
    
    # Loss function and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=2.5e-4)
    best_nme = 99.0

    for epoch in range(max_epoch):
        print("=============Epoch %i================" %(epoch))
    
        # Adjust Learning Rate based on epoch number
        if epoch in list(lr.keys()):
            adjust_lr(optimizer, lr[epoch])
            
        # Train 
        for i_batch, sample_batch in enumerate(train_dataloader,0):
            net.zero_grad() # Zero gradients after each batch
            loss = 0
            
            inputs, targets = sample_batch['image'], sample_batch['heatmaps']
            inputs = inputs.to(device); targets = targets.to(device)
            outputs = net(inputs)
            
            for o in outputs:
                loss += criterion(o, targets)
            loss.backward()
            optimizer.step()
            
            # Print training loss every N batches
            if (i_batch % print_every==0):
                print("Batch %i/%i;	Loss: %.4f" %(i_batch, num_batches, loss.item()))
        
        ## Evaluation at the end of epoch
        net = net.eval()
        current_nme = 0

        with torch.no_grad():
            for i_test, test_batch in enumerate(test_dataloader, 0):
                inputs, landmarks, boxes = test_batch['image'], \
                     test_batch['landmarks'], test_batch['bbox']
                inputs = inputs.to(device); boxes= boxes.to(device); landmarks = landmarks.to(device)
                outputs = net(inputs)
                nme = NME(outputs[-1], landmarks, boxes)
                current_nme += nme

        current_nme /= len(test_dataloader)
        print("Test NME: %.8f" %(current_nme))

        # Save model if it is the best thus far
        if current_nme < best_nme:
            best_nme = current_nme
            torch.save(net.state_dict(), "ckpt_epoch_"+str(epoch))
        net = net.train()

            	
def main():
    train()


if __name__=='__main__':
    main()	
