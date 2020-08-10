import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.inception import inception_v3
from tqdm import tqdm
from scipy.stats import entropy


def compute_inception_score(G, device):
    """compute the inception score"""
    batch_size = 64
    N = int(50000 / batch_size)
    splits = 10
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    def get_pred(x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = inception_model(x)
        return torch.softmax(x, -1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N * batch_size, 1000))

    for i in range(N):
        noise = torch.randn(batch_size, 128, 1, 1).to(device)
        with torch.no_grad():
            batch = G(noise)
        batch_size_i = batch.size()[0]
        pr = get_pred(batch)
        preds[i * batch_size:i * batch_size + batch_size_i] = pr

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def imshow(img):
    """transform the image before the plt.show..."""
    npimg = (img.cpu()).numpy()
    return np.transpose(npimg, (1, 2, 0))


def weights_init_normal(m):
    """ init the weight of the network"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
