import torch
from models.stylegan2.model import Generator1, Encoder, MappingNetwork
from torchvision.utils import make_grid
from torchvision import transforms, utils
from torch.utils import data

import random
import math

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from matplotlib import pyplot as plt
from PIL import Image




# libs for visualization

import umap
from io import BytesIO
from PIL import Image
import base64

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10
from bokeh.palettes import RdBu3
from bokeh.models import Select
from bokeh.layouts import column
from bokeh.models.callbacks import CustomJS

if __name__ == '__main__':
    pass