# This parser file is for juypter notebook.
import os
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
import torch
import torch.optim as optim

def get_parser():
    parser = edict()
    