from __future__ import print_function
from PIL import Image
import argparse
from math import log10
from os.path import exists, join, basename, splitext
from os import makedirs, remove ,listdir
import time
import pandas
import random
import time
import os
import shutil
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net_sasr_fourx4 import Net_spatial
from dataset_four import DatasetFromFolder
import numpy as np

def reconstruction_loss(X, Y):
	# L1 Charbonnier loss
	eps = 1e-6
	diff = torch.add(X, -Y)
	error = torch.sqrt(diff * diff + eps)
	loss = torch.sum(error) / torch.numel(error)
	return loss

def gradient(pred):
	D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
	D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
	return D_dx, D_dy

def epi_loss(pred, label):
	def lf2epi(lf):
		N, an2, h, w = lf.shape
		an = int(math.sqrt(an2))
		epi_h = lf.view(N, an, an, h, w).permute(0, 1, 3, 2, 4).contiguous().view(-1, 1, an, w)
		epi_v = lf.view(N, an, an, h, w).permute(0, 2, 4, 1, 3).contiguous().view(-1, 1, an, h)
		return epi_h, epi_v

	epi_h_pred, epi_v_pred = lf2epi(pred)
	dx_h_pred, dy_h_pred = gradient(epi_h_pred)
	dx_v_pred, dy_v_pred = gradient(epi_v_pred)

	epi_h_label, epi_v_label = lf2epi(label)
	dx_h_label, dy_h_label = gradient(epi_h_label)
	dx_v_label, dy_v_label = gradient(epi_v_label)

	return reconstruction_loss(dx_h_pred, dx_h_label) + reconstruction_loss(dy_h_pred,dy_h_label) + \
		   reconstruction_loss(dx_v_pred,dx_v_label) + reconstruction_loss(dy_v_pred, dy_v_label)

class data_prefetcher():
	def __init__(self, loader):
		self.loader = iter(loader)
		self.preload()
	def preload(self):
		try:
			self.next_data = next(self.loader)
		except StopIteration:
			self.next_data = None
			return
	def next(self):
		data = self.next_data
		self.preload()
		return data

def checkpoint(model,file_dir,file_name):
	if not exists(file_dir):
		makedirs(file_dir)
	model_out_path = join(file_dir,file_name)
	torch.save({'state_dict': model.state_dict()}, model_out_path)
