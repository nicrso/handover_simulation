from typing import Dict, List, Tuple

import pathlib
import numpy as np
import cv2
import torch
import os
import pandas as pd


def get_splits(type) -> Dict[str, List[str]]:
    """
    Read read training/testing split.
    """
    if type == 'handoff':
        csv_path = os.path.join(os.path.dirname(__file__),'assets/handoff','split.csv')
    elif type == 'use':
        csv_path = os.path.join(os.path.dirname(__file__),'assets/use','split.csv')
    else: 
        csv_path = os.path.join(os.path.dirname(__file__),'assets/ycb_objects','split.csv')
    df = pd.read_csv(csv_path, header=0)
    splits = {
        'train': list(df['Name'].loc[df['Split'] == 'Train']),
        'test': list(df['Name'].loc[pd.isnull(df['Split'])])
    }
    return splits 