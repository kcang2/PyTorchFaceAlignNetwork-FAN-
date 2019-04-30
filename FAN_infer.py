import torch
from models import FAN
from face_dataset import create_dataloader
import random
import numpy as np
from face_utils import *


def infer():
    test_dataloader = create_dataloader(root='.', batch_size=32, is_train=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = FAN(); net.eval(); net.to(device)
    net.load_state_dict(torch.load("ckpt_epoch_28"))

    running_nme = 0.0

    for sample in test_dataloader:
        images, targets, t_hm, boxes = sample['image'], sample['landmarks'], \
            sample['heatmaps'], sample['bbox']
        
        images = images.to(device); targets = targets.to(device); boxes = boxes.to(device)

        # DO NOT ACCUMULATE GRADIENTS DURING FORWARD PASS
        with torch.no_grad():
            preds = net(images)

        # break  # if we just want an image to illustrate

        # Compute batch NME
        running_nme += NME(preds[-1], targets, boxes)

    # Test data NME
    print("Evaluation NME:")
    print(running_nme/len(test_dataloader))

#    For Illustration Purposes
#    Convert predicted heatmaps to landmarks, scale from heatmap dimension to bounding box dimension
#    move image to CPU, convert to ndarray, transpose from NCHW to NHWC
#    Other inputs must be moved to CPU as well

#    preds = get_pts(preds[-1], ((boxes[:,1]-boxes[:,0]+1)/t_hm.shape[2]).unsqueeze(dim=1))
#    draw( images[0].to("cpu").numpy().transpose((1,2,0)),\
#            preds[0].to("cpu"), targets[0].to("cpu"),\
#            (boxes[0,1]-boxes[0,0]+1).cpu().numpy().astype(np.int32) )


def main():
    infer()


if __name__ == "__main__":
    main()
