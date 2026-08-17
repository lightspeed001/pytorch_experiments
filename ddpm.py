import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# Hyperparameters

BATCH_SIZE = 128
IMAGE_SIZE= 32
T = 1000 # Total timestamps
LEARNING_RATE = 2e-4
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is.available() else "cpu"


#Diffusion parameters

