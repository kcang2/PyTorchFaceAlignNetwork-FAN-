import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np

def NME(preds, targets, boxes):
    ## Compute Normalized Mean Error using predicted heatmaps, ground-truth landmarks and bounding boxes ##
    ## preds: N x L x H x H; N = batch size; L = number of landmarks; H = heatmap dimension
    ## targets: N x L x 2; [x y]
    ## boxes: N x 2 x 2; [top-left bottom-right]; [x y]

    D = torch.squeeze(boxes[:, 1] - boxes[:, 0] + 1, dim=1)  # N x 2
    w, h = D[:, 0], D[:, 1] 
    RAs = torch.sqrt(w*h)  # Square root of box area, N
    if len(preds.shape)>3 :
        ratios = D.unsqueeze(dim=1)/preds.shape[2]  # scale, from 64 to w, h
        preds = get_pts(preds, ratios)  # convert heatmaps to landmarks
    mse = torch.sqrt((preds - targets)**2).sum(2) # L2-norm / sum-squared error; N x L
    nme = torch.mean(mse/RAs.reshape(-1,1)) # N x L divide N
    return float(nme) 


def get_pts(heatmaps, ratio):
    ## Get landmarks from heatmaps with scaling ratios ##
    ## heatmaps: N x L x H x H ##
    ## ratio: N ##

    # Get x and y of landmarks
    lmx = torch.argmax(torch.max(heatmaps, dim=2)[0], dim=2).type(torch.cuda.FloatTensor)
    lmy = torch.argmax(torch.max(heatmaps, dim=3)[0], dim=2).type(torch.cuda.FloatTensor)

    # Stack them and scale with ratios
    landmarks = torch.stack((lmx,lmy), dim=2)
    landmarks *= ratio
    return landmarks


def draw(im, lmp, lmt, new_size=None):
    ## Draw out image with predicted and ground-truth landmarks ##
    ## Possibly resize the image if needed ##

    # Resize image if needed
    if new_size is not None:
        im = resize(im, new_size)

    # matplotlib requires ints to display image, not float
    im = im.astype(np.int32)
    fig,ax = plt.subplots(1)
    plt.scatter(lmp[:, 0], lmp[:, 1], s=10, marker='.', c='r') # Predictions in Red
    plt.scatter(lmt[:, 0], lmt[:, 1], s=10, marker='.', c='b') # Truths in Blue
    ax.imshow(im)
    plt.show()

