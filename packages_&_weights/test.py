"""
Script to test imports and loading of model weights
"""

import logging
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import pandas as pd
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time
import math
import random
from tqdm import tqdm
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import cv2
import yaml
import torch
import seaborn as sns
import albumentations as A

from configs import TrainingConfig
from utils.rf_detr_extract_utils import RFDETRExtractor
from utils.preprocessing_utils import PreprocessingUtils
from utils.data_logging_utils import DataLogger
from utils.optuna_utils import OptunaTrialManager

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
os.environ["ALBUMENTATIONS_DISABLE"] = "1"
os.environ["USE_LIBUV"] = "0" # Disable libuv to prevent potential issues with subprocesses in Windows environments
from rfdetr import RFDETRNano
model = RFDETRNano(pretrain_weights="rf-detr-nano.pth")
model = None