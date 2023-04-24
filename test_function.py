from __future__ import print_function
from os.path import exists, join, basename
from os import makedirs, remove, makedirs, path
import argparse
import torch
import pandas
import random
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor, Resize
from glob import glob
import numpy as np
from os import listdir
from math import log10
import cv2
from skimage import color,io
from thop import profile
from net_sasr import Net
from einops import rearrange

def test(views,LF,opt):
	cuda = True
	ds = opt.ang_upscale
	rand_num=opt.rand_num
	angout=(opt.angin-1)*opt.ang_upscale+1
	spa_scale=opt.spa_upscale
	angin_uint=opt.angin_uint
	angout_uint=(angin_uint-1)*ds+1
	model_spatial = Net(rand_num,spa_scale,ds)
	checkpoint1 = torch.load(opt.model_dir+LF+'/'+LF+'_spa_ang_max_in50_psnr.pth.tar')
	model_spatial.load_state_dict(checkpoint1['state_dict'])
	if cuda:
		model_spatial = model_spatial.cuda()
	input1=torch.randn(1, 1,  32, 32).cuda()
	input_r=[torch.randn(1, 1,  32, 32).cuda() for i in range(4)]
	flops, params = profile(model_spatial, inputs=(input1,input1,input1,input1,input_r,input_r))
	print('spanet的FLOPs：' + str(flops*16 / 1000 ** 3) + 'G')
	print('model_spa Total: {}M'.format(params / (10 ** 6)))

	total_num = sum(p.numel() for p in model_spatial.parameters())
	trainable_num = sum(p.numel() for p in model_spatial.parameters() if p.requires_grad)
	print('model_spatial_angular Total: {}M'.format(total_num / (10 ** 6)), 'Trainable: {}M'.format(trainable_num / (10 ** 6)))
	view_num = []
	for i in range(0, angout-1, ds):
		for j in range(0, angout-1, ds):
				view_num.append([i, j])
	a = np.shape(views)
	pre_view = np.zeros((a[0],a[1],a[2]*spa_scale,a[3]*spa_scale),dtype=np.float32)
	num_block=(angout-angout_uint)//(angout_uint-1)+1
	pre_view_block=torch.ones((num_block,num_block,angout_uint,angout_uint,a[2]*spa_scale,a[3]*spa_scale))
	input=[[0,0] for i in range(angin_uint)]
	view_list = []
	overlap_list = [i for i in range(angout_uint - 1, angout - 1, angout_uint - 1)]
	with torch.no_grad():
		for ii in range(len(view_num)):
			hsta,wsta=view_num[ii]
			for i in range(angin_uint):
				for j in range(angin_uint):
					img=views[hsta +i*ds,wsta +j*ds,:,:].copy()
					input[i][j]=Variable(ToTensor()(img[:,:,np.newaxis])).view(1, -1, img.shape[0], img.shape[1])
					if cuda:
						input[i][j]=input[i][j].cuda()
			b = [i for i in range(0,angout,ds)]
			input_random=[]
			random_list = []
			order_list=[[hsta,wsta],[hsta+ds,wsta],[hsta,wsta+ds],[hsta+ds,wsta+ds]]
			while len(random_list) != opt.rand_num:
				p = random.choice(b)
				q = random.choice(b)
				if [p, q] not in random_list:
					if ds==4:
						if len(random_list) == opt.rand_num-1:
							if random_list[0] in order_list and random_list[1] in order_list and random_list[2] in order_list:
								if [p, q] in order_list:
									continue
					random_list.append([p, q])
					input_random.append(views[p,q,:,:].copy())
			input_random_sobel = []
			for i in range(rand_num):
				input_random_sobel.append(mycanny(input_random[i]))
				input_random[i] = Variable(ToTensor()(input_random[i][:, :, np.newaxis])).view(1, -1, input_random[i].shape[0], input_random[i].shape[1])
				input_random_sobel[i] = Variable(ToTensor()(input_random_sobel[i][:, :, np.newaxis])).view(1, -1, input_random_sobel[i].shape[0], input_random_sobel[i].shape[1])
				if cuda:
					input_random[i]=input_random[i].cuda()
					input_random_sobel[i] = input_random_sobel[i].cuda()
			HR = model_spatial(input[0][0],input[0][1],input[1][0],input[1][1],input_random,input_random_sobel)
			HR = HR.cpu().data[0]
			pre_view_block[ii//num_block, ii%num_block] = torch.clamp(HR.view(angout_uint,angout_uint,HR.shape[1],-1), 16 / 255, 235 / 255)

		epi_hor_block = rearrange(pre_view_block, "b1 b2 a1 a2 H W -> (b1 b2 a1 H) 1 a2 W")
		epi_ver_block = rearrange(pre_view_block, "b1 b2 a1 a2 H W -> (b1 b2 a2 W) 1 a1 H")
		nn_unfold=torch.nn.Unfold(angout_uint,1,0,1)
		epi_hor_block = nn_unfold(epi_hor_block)
		epi_ver_block = nn_unfold(epi_ver_block)
		epi_hor_block = rearrange(epi_hor_block, "(b1 b2 a1 H) k l -> b1 b2 a1 H l k",b1=num_block,b2=num_block,a1=angout_uint)
		epi_ver_block = rearrange(epi_ver_block, "(b1 b2 a2 W) k l -> b1 b2 a2 l W k",b1=num_block,b2=num_block,a2=angout_uint)
		epi_hor_block,epi_ver_block=epi_hor_block.numpy(),epi_ver_block.numpy()
		epi_hor_block=np.std(epi_hor_block, axis=-1)
		epi_ver_block=np.std(epi_ver_block, axis=-1)
		pre_view_block=pre_view_block.numpy()
		for i in range(num_block):
			for j in range(num_block):
				for m in range(angout_uint):
					for n in range(angout_uint):
						k1,k2=view_num[i*num_block+j]
						k1,k2=k1+m,k2+n
						if [k1,k2] not in view_list:
							view_list.append([k1,k2])
							if k1 not in overlap_list and k2 not in overlap_list:
								pre_view[k1,k2]=pre_view_block[i,j,m,n]
							if k1 in overlap_list and k2 not in overlap_list:
								pre_view[k1,k2]=combin_twoview(pre_view_block,epi_ver_block,i,j,m,n,opt,'vertical')
							if k1 not in overlap_list and k2 in overlap_list:
								pre_view[k1, k2] = combin_twoview(pre_view_block,epi_hor_block,i, j, m, n,opt,'horizontal')
							if k1 in overlap_list and k2 in overlap_list:
								pre_view[k1, k2]=combin_fourview(pre_view_block,epi_hor_block,epi_ver_block,i,j,m,n,opt)
	return pre_view

def mycanny(img):
	img1 = (img * 255).astype(np.uint8)
	dst=cv2.Canny(img1,50,100)
	kernel = np.ones((5, 5), np.uint8)
	dst = cv2.dilate(dst, kernel, 1)
	dst = dst.astype(np.float32)
	return dst

def combin_twoview(pre_view_block,epi_block,i,j,m,n,opt,dire):
	img1 = pre_view_block[i, j, m, n]
	if dire == 'horizontal':
		img2 = pre_view_block[i, j+1, m, 0]
	if dire == 'vertical':
		img2 = pre_view_block[i + 1, j, 0, n]
	weight = np.ones_like(img1)
	angout_uint=(opt.angin_uint-1)*opt.ang_upscale+1
	weight[:, :] = 0.5
	dif = np.absolute(img1-img2)
	larger_dif_list = np.where(dif > 0.02, 1, 0)
	smaller_dif_list = np.where(dif > 0.02, 0, 0.5)
	if dire == 'horizontal':
		weight[:,angout_uint//2:-(angout_uint//2)] = 1 - epi_block[i,j,m]/(epi_block[i,j,m]+epi_block[i,j+1,m]+1e-10)
	if dire == 'vertical':
		weight[angout_uint//2:-(angout_uint//2),:] = 1 - epi_block[i,j,n]/(epi_block[i,j,n]+epi_block[i+1,j,n]+1e-10)
	weight=np.multiply(larger_dif_list,weight)+smaller_dif_list
	com_imgae = np.multiply(weight, img1) + np.multiply(1 - weight, img2)

	return com_imgae

def combin_fourview(pre_view_block,epi_hor_block,epi_ver_block,i,j,m,n,opt):
	img1=combin_twoview(pre_view_block,epi_hor_block,i,j,m,n,opt,'horizontal')
	img2=combin_twoview(pre_view_block,epi_ver_block,i,j,m,n,opt,'vertical')
	img3=combin_twoview(pre_view_block,epi_hor_block,i+1,j,0,n,opt,'horizontal')
	img4=combin_twoview(pre_view_block,epi_ver_block,i,j+1,m,0,opt,'vertical')
	com_image=(img1+img2+img3+img4)/4
	return com_image
