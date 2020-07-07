#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0][1])
    for k in w_avg.keys():
        w_avg[k] -= w[0][1][k]

    for k in w_avg.keys():
        count = 0
        for i in range(len(w)):
            if w[i][0] == 1:
                count+=1
                w_avg[k] += w[i][1][k]
        w_avg[k] = torch.div(w_avg[k], count)
    return w_avg
