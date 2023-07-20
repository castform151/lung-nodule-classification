from kflod import kfold
import numpy as np
import random
from torch.optim.adam import Adam
from resnet_attn import *
from preprocessing import get_dataset3d
import sys
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def reset_rand():
    seed = 1000
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def expLocalGlobal(data_path):
    reset_rand()

    def model_opt():
        model = LocalGlobalNetwork()
        optm = Adam(model.parameters())
        return model, optm

    kfold(data_path,
          256,
          1,
          model_optimizer=model_opt,
          loss=nn.BCELoss(),
          name='LocalGlobalNetwork',
          device='cpu',
          deterministic=True
          )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error, we expect one argument')

    else:
        data_path = sys.argv[1]
        expLocalGlobal(data_path)
