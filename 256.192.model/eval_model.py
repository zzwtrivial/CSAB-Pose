import argparse
import json
import sys
from datetime import datetime

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
from pycocotools.coco_custom import COCO
from pycocotools.cocoeval_custom import COCOeval
from tqdm import tqdm

from dataloader.LMDB_train_ll import LmdbDataset
from networks import network
from test_config_all import cfg_test_all
# from allconfig import cfg
from train_ll_cpn_config import cfg
from utils.evaluation import AverageMeter
from utils.imutils import *
from utils.logger import Logger
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join

