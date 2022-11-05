import numpy as np
import time
import os
import argparse
import shutil
from tqdm import tqdm

checkpoints_path = "/remote-home/gengyu/MLN-main/checkpoints"
for root, dirs, files in os.walk(checkpoints_path, topdown=False):
    if not os.listdir(root):
        os.rmdir(root)

print("i am green!")