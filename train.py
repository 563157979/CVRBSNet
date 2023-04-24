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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net_sasr import Net
from dataset import DatasetFromFolder
from utils import *
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--dataset', type=str, default="../../dataset/mix_data/datax4/", help='input dataset dir')
parser.add_argument('--lr', type=float, default=0.0015, help='Learning Rate. Default=0.001')
parser.add_argument('--nocuda', action='store_false', help='use cuda?')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--up_factor', type=int, default=4, help='spatial and angular upsample factor')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--aug', type=int, default=1, help='data augmentation. Default=1')
parser.add_argument('--testpath', type=str, default='result', help='train_test-result')
parser.add_argument("--epi", type=float, default=0.2, help="epi loss")
parser.add_argument('--rand_num', type=int, default=4, help='random view number')

os.environ['CUDA_VISIBLE_DEVICES']='0'

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

def ang_spa_train(epoch,LF_order,load_data,model_ang_spa):
	LF_loss_ang_spa = 0
	iteration = 0
	prefetcher = data_prefetcher(load_data)
	batch = prefetcher.next()
	while batch is not None:
		iteration += 1
		sobelimg = []
		input_r=[]
		input_1, input_2, input_3, input_4, target= Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), Variable(batch[4])
		for i in range(len(batch[5])):
			input_r.append(Variable(batch[5][i]))
			sobelimg.append(Variable(batch[6][i]))
			sobelimg[i] = sobelimg[i].cuda()
			input_r[i]=input_r[i].cuda()
		if cuda:
			input_1 = input_1.cuda()
			input_2 = input_2.cuda()
			input_3 = input_3.cuda()
			input_4=input_4.cuda()
			target = target.cuda()
		HR = model_ang_spa(input_1, input_2,input_3, input_4,input_r,sobelimg)
		loss_ang_spa = criterion(HR, target)+opt.epi *epi_loss(HR, target)

		LF_loss_ang_spa += loss_ang_spa.item()
		optimizer_ang_spa.zero_grad()
		loss_ang_spa.backward()
		optimizer_ang_spa.step()
		batch = prefetcher.next()
	loss_ang_spa_avg = LF_loss_ang_spa / len(load_data)
	return loss_ang_spa_avg

def ang_spa_valid(load_data,model_ang_spa):
	LF_ang_spa_psnr = 0
	for iter,batch in enumerate(load_data):
		sobelimg=[]
		input_r=[]
		if iter==0:
			batch[4]=batch[4].squeeze()
			target=batch[4]
			img_size =batch[4].size()
			LFHR = torch.ones(img_size)
			if cuda:
				LFHR=LFHR.cuda()
				target=target.cuda()
		with torch.no_grad():
			input_1, input_2, input_3, input_4= Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
			for i in range(len(batch[5])):
				input_r.append(Variable(batch[5][i]))
				sobelimg.append(Variable(batch[6][i]))
				sobelimg[i] = sobelimg[i].cuda()
				input_r[i] = input_r[i].cuda()
			if cuda:
				input_1 = input_1.cuda()
				input_2 = input_2.cuda()
				input_3 = input_3.cuda()
				input_4 = input_4.cuda()
			HR = model_ang_spa(input_1, input_2, input_3, input_4,input_r,sobelimg)
			rstart = iter // 4 * opt.up_factor
			cstart = iter % 4 * opt.up_factor
			LFHR[rstart:rstart+opt.up_factor+1,cstart:cstart+opt.up_factor+1,:,:]=HR.view(opt.up_factor+1,opt.up_factor+1,img_size[2],img_size[3])
	for i in range(img_size[0]):
		for j in range(img_size[1]):
			mse = criterion(LFHR[i,j,:,:], target[i,j,:,:])
			psnr = 10 * log10(1 / mse.item())
			LF_ang_spa_psnr += psnr
	psnr_avg=LF_ang_spa_psnr/img_size[0]/img_size[1]
	return psnr_avg

def main(opt):
	cuda = opt.nocuda
	if cuda and not torch.cuda.is_available():
		raise Exception("No GPU found, please run without --cuda")
	# 设置随机种子
	torch.manual_seed(opt.seed)
	if cuda:
		torch.cuda.manual_seed(opt.seed)

	datasets_dir=opt.dataset
	datasets=listdir(datasets_dir)
	datasets.sort()

	criterion = nn.MSELoss()
	if cuda:
		criterion = criterion.cuda()
	loss_ang_spa_alllist=[]
	psnr_ang_spa_alllist=[]
	average_max_psnr_alllist=[]
	rand_num=opt.rand_num
	for dataset in datasets:
		tes_dir=datasets_dir+dataset
		test_file=listdir(tes_dir)
		test_file.sort()
		test_LF_num=len(test_file)
		average_max_psnr_list = []
		average_max_psnr_list.append(dataset)
		max_psnr_50,max_psnr_40=[],[]
		print(dataset)
		epochnum = 8000
		for LF_order in range(test_LF_num): #把测试数据按不同的LF加载
			loss_ang_spa_list = []
			psnr_ang_spa_list = []
			time1=time.time()
			train_set = DatasetFromFolder(tes_dir, LF_order, 'training',opt.aug,epochnum,rand_num)  # train_set为DatasetFromFolder类的对象数组
			valid_set = DatasetFromFolder(tes_dir, LF_order, 'test',0,1,rand_num)
			test_tra_loader=DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
			test_val_loader=DataLoader(dataset=valid_set, num_workers=0, batch_size=opt.testBatchSize, shuffle=False)
			print('===> Building model')
			file_dir = './{}/CSV'.format(opt.testpath)
			if not exists(file_dir):
				makedirs(file_dir)
			lr=opt.lr
		# -------------------------ang_spa_train--------------------------------------
			model_ang_spa = Net(rand_num,opt.up_factor,opt.up_factor)  # 定义超分辨网络模型
			if cuda:
				model_ang_spa=model_ang_spa.cuda()
			total_num = sum(p.numel() for p in model_ang_spa.parameters())
			trainable_num = sum(p.numel() for p in model_ang_spa.parameters() if p.requires_grad)
			print('model_spatial_angular Total: {}M'.format(total_num / (10 ** 6)),'Trainable: {}M'.format(trainable_num / (10 ** 6)))
			optimizer_ang_spa = optim.Adam(model_ang_spa.parameters(), lr,betas=(0.9, 0.999),eps=1e-08)
			scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer_ang_spa, [40], gamma=0.5, last_epoch=-1)
			loss_ang_spa_list.append(test_file[LF_order])
			psnr_ang_spa_list.append(test_file[LF_order])
			for epoch in range(1,opt.nEpochs+1):
				time3=time.time()
				ang_spa_loss = ang_spa_train(epoch, LF_order, test_tra_loader,model_ang_spa)
				loss_ang_spa_list.append(ang_spa_loss)
				if epoch >=0:
					model_name = test_file[LF_order]+"_spa_ang_epoch{}.pth.tar".format(epoch)
					model_dir = "./{}/model/model_spa_ang/".format(opt.testpath)+test_file[LF_order]
					checkpoint(model_ang_spa, model_dir, model_name)
				if epoch >=0:
					ang_spa_psnr = ang_spa_valid(test_val_loader,model_ang_spa)
					print("===> LF: " + test_file[LF_order] + " epoch[{}] ang_spa_PSNR: {:.4f} dB".format(epoch , ang_spa_psnr))
					psnr_ang_spa_list.append(ang_spa_psnr)
				scheduler.step()
				print(optimizer_ang_spa.state_dict()['param_groups'][0]['lr'])
				time4=time.time()
				print("LF-{} epoch{} train+test time: {}".format(test_file[LF_order],epoch,time4-time3))
			time2 = time.time()
			psnr_ang_spa_list.append(time2-time1)
			max_psnr_num_50 = np.argmax(psnr_ang_spa_list[1:-1])+1
			max_psnr_num_40 = np.argmax(psnr_ang_spa_list[1:-11])+1
			max_psnr_50.append(np.max(psnr_ang_spa_list[1:-1]))
			max_psnr_40.append(np.max(psnr_ang_spa_list[1:-11]))
			psnr_ang_spa_list.extend([max_psnr_num_50, np.max(psnr_ang_spa_list[1:-1]), max_psnr_num_40, np.max(psnr_ang_spa_list[1:-11])])
			loss_ang_spa_alllist.append(loss_ang_spa_list)
			psnr_ang_spa_alllist.append(psnr_ang_spa_list)
			old_model_dir = "./{}/model/model_spa_ang/".format(opt.testpath) + test_file[LF_order] + '/' + test_file[LF_order] + "_spa_ang_epoch{}.pth.tar".format(max_psnr_num_50)
			new_model_dir = "./{}/model/model_spa_ang/".format(opt.testpath) + test_file[LF_order] + '/'
			shutil.copy(old_model_dir, new_model_dir + test_file[LF_order] + '_spa_ang_max_in50_psnr.pth.tar')
			old_model_dir ="./{}/model/model_spa_ang/".format(opt.testpath)+test_file[LF_order]+'/'+test_file[LF_order]+"_spa_ang_epoch{}.pth.tar".format(max_psnr_num_40)
			new_model_dir="./{}/model/model_spa_ang/".format(opt.testpath)+test_file[LF_order]+'/'
			shutil.copy(old_model_dir,new_model_dir+test_file[LF_order]+'_spa_ang_max_in40_psnr.pth.tar')
			print("LF-{} train+test time: {}".format(test_file[LF_order],time2 - time1))
			p = pandas.DataFrame(loss_ang_spa_alllist)
			p.to_csv('./{}/CSV/loss_ang_spa_test_list.csv'.format(opt.testpath))
			p = pandas.DataFrame(psnr_ang_spa_alllist)
			p.to_csv('./{}/CSV/psnr_ang_spa_list.csv'.format(opt.testpath))
		average_max_psnr_list.extend([np.average(max_psnr_50),np.average(max_psnr_40)])
		average_max_psnr_alllist.append(average_max_psnr_list)
		p = pandas.DataFrame(average_max_psnr_alllist)
		p.to_csv('./{}/CSV/average_max_set_psnr.csv'.format(opt.testpath))

if __name__ == '__main__':
	opt = parser.parse_args()
	print(opt)
	main(opt)
