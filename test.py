# from kflod import kfold
import numpy as np
import random
from torch.optim.adam import Adam
from resnet_attn import *
from preprocessing import get_dataset3d, get_dataset
import sys
import torch
from sklearn import metrics
from trainer import Trainer
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# def summery(data):
#     n = 0.0
#     s_dist = 0
#     for dist in data:
#         s_dist += torch.sum(dist)
#         n += len(dist)
#     return s_dist.float() / n


def calc_accuracy(x, y):
    x_th = (x > 0.5).long()
    matches = x_th == y.long()
    return matches


def get_metrics(target, pred):
    prec, recall, _, _ = metrics.precision_recall_fscore_support(
        target, pred > 0.5, average='binary')
    fpr, tpr, thresholds = metrics.roc_curve(target, pred)
    auc = metrics.auc(fpr, tpr)
    return prec, recall, auc


model = LocalGlobalNetwork()
model.load_state_dict(torch.load("results/localglobal_group26.pth"))
model.eval()

trset, testset = get_dataset("./dataset/")
tr = Trainer(
    trset,
    testset,
    256,
    50,
    model,
    Adam(model.parameters()),
    nn.BCELoss(),
    "test",
    device='cpu',
)
# valid_acc = tr.validate()
# test_dist = summery(valid_acc)

pred, target = tr.predict()
prec, recall, auc = get_metrics(target, pred)

matches = calc_accuracy(pred, target)
acc = matches.float().mean()

msg = f'accuray: {acc}, AUC: {auc}, precession: {prec}, Recall: {recall}'
print(msg)
