from __future__ import print_function
from os.path import exists, join, basename
from os import makedirs, remove,  path
import argparse
import torch
import pandas
import random
import shutil
import time
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor, Resize
import numpy as np
from os import listdir
from math import log10
from skimage.metrics import structural_similarity
from test_function import test
import cv2
from skimage import color,io
Image.MAX_IMAGE_PIXELS = 200000000
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--save_img', type=int, default=0, help='is it to save result image?')
parser.add_argument('--spa_upscale', type=int, default=2, help='spatial upscale factor')
parser.add_argument('--ang_upscale', type=int, default=2, help='angular upscale factor')
parser.add_argument('--result_dir', type=str, default='resultx2', help='SR result saving dir')
parser.add_argument('--data_dir', type=str, default="../../../dataset/mix_data/valid_test_all", help='dataset dir')
parser.add_argument('--model_dir', type=str, default="../../lunwenx2/model/model_spa_ang/", help='model dir')
parser.add_argument('--rand_num', type=int, default=4, help='random view number in EBSSE')
parser.add_argument('--angin', type=int, default=5, help='input LF angular resolution')
parser.add_argument('--angin_uint', type=int, default=2, help='input uint angular resolution')
opt = parser.parse_args()
print(opt)
##-------------------------lf_compare---------------------------------
result_dir=opt.result_dir
data_dir=opt.data_dir
angw,angh=(opt.angin-1)*opt.ang_upscale+1,(opt.angin-1)*opt.ang_upscale+1
save_img=opt.save_img
spa_upscale=opt.spa_upscale
ang_upscale=opt.ang_upscale
sets=listdir(data_dir)
sets.sort()
f=open('../../../dataset/mix_data/chosen_image.txt')
name_list=f.readlines()
lf_name_list=[i.rstrip() for i in name_list]

view_factor = [['光场图片名称','PSNR','SSIM','子图像名']]
lf_factor=[['光场图片名称','ALL_PSNR_MAX','最大PSNR视角位置','ALL_SSIM_MAX','最大SSIM视角位置','平均PSNR','平均SSIM','最小PSNR','最小SSIM','测试耗时-s']]
set_factor=[['数据集名称','ALL_PSNR','ALL_SSIM']]
epi_factor = [['光场图片名称','PSNR','SSIM','子图像名']]
lf_epi_factor=[['光场图片名称','ALL_PSNR_MAX','最大PSNR视角位置','ALL_SSIM_MAX','最大SSIM视角位置','平均PSNR','平均SSIM','最小PSNR','最小SSIM']]
set_epi_factor=[['数据集名称','ALL_PSNR','ALL_SSIM']]
save_imgname=['origami','sideboard','Tarot Cards and Crystal Ball (small angular extent)_rectified','lego kinghts_rectified']
save_setname=["stanford_gantry"]
if not path.exists('./{}/test_result'.format(result_dir)):
	makedirs('./{}/test_result'.format(result_dir))
p = pandas.DataFrame(view_factor)
p.to_csv('./{}/test_result/test_view_psnr_ssim.csv'.format(result_dir))
for set in sets:
	set_dir=data_dir+'/'+set
	lfs=listdir(set_dir)
	lfs.sort()
	set_psnr = []
	set_ssim = []
	set_epi_psnr = []
	set_epi_ssim = []
	print('##############################',set)
	for lf in lfs:
		if lf in lf_name_list:
			org_lf_dir=set_dir+'/'+lf
			if save_img and ((lf in save_imgname) or (set in save_setname)):
				save_view_path=result_dir +'/test_view_result_image/'+ set+"/"+lf
				if not path.exists(save_view_path):
					makedirs(save_view_path)
				save_epi_path = result_dir +'/test_epi_result_image/'+ set + "/" + lf
				if not path.exists(save_epi_path):
					makedirs(save_epi_path)
			view_psnr = []
			view_ssim = []
			epi_psnr = []
			epi_ssim = []

			view=io.imread(org_lf_dir + '/1_1.png')
			[h,w,c]=view.shape
			h=h//spa_upscale*spa_upscale
			w=w//spa_upscale*spa_upscale
			lf_img=np.ones((angh,angw,h,w,c),dtype=np.float32)
			lr_img=np.ones((angh,angw,h//spa_upscale,w//spa_upscale),dtype=np.float32)
			out=np.ones((angh,angw,h,w,c),dtype=np.uint8)
			for i in range(angh):
				for j in range(angw):
					view=io.imread(org_lf_dir + '/{}_{}.png'.format(i + 1, j + 1))
					view=view[:h,:w,:]
					lf_img[i,j,:,:,:]=(color.rgb2ycbcr(view)/255).astype(np.float32)
					lr_img[i,j,:,:]=cv2.resize(lf_img[i,j,:,:,0],(0,0),fx=1/spa_upscale,fy=1/spa_upscale,interpolation=cv2.INTER_CUBIC)
			starttime = time.time()
			views_pre=test(lr_img,lf,opt)
			endtime = time.time()
			lftime = endtime - starttime
			for i in range(angh):
				for j in range(angw):
					view_pre = views_pre[i, j]
					view_org = lf_img[i, j]
					a = list(view_org.shape)
					if save_img and ((lf in save_imgname) or (set in save_setname)):
						print(lf,set)
						out[i, j] = process(view_pre, view_org[:, :, 1], view_org[:, :, 2], a)
						io.imsave(save_view_path + '/' + '{}_{}.png'.format(i + 1, j + 1), out[i, j])
					mse = np.mean((view_pre - view_org[:, :, 0]) ** 2)
					psnr = 10 * log10(1 / mse)
					ssim = structural_similarity(view_pre, view_org[:, :, 0])
					view_psnr.append(psnr)
					view_ssim.append(ssim)
					view_factor.append([lf, psnr, ssim, '{}_{}.png'.format(i + 1, j + 1)])
					####获取epi图像，并计算每个epi的PSNR和SSIM
			for i in range(angh):
				for j in range(h):
					epi1_rgb = out[i, :, j, :, :]
					if save_img and ((lf in save_imgname) or (set in save_setname)):
						io.imsave(save_epi_path + '/{}_{}.png'.format(i + 1, j + 1), epi1_rgb)
					epi1 = views_pre[i, :, j, :]
					epi2 = lf_img[i, :, j, :, 0]
					mse = np.mean((epi1 - epi2) ** 2)
					psnr = 10 * log10(1 / mse)
					ssim = structural_similarity(epi2, epi1)
					epi_psnr.append(psnr)
					epi_ssim.append(ssim)
					epi_factor.append([lf, psnr, ssim, '{}_{}.png'.format(i + 1, j + 1)])

			lf_factor.append([lf, np.max(view_psnr),'{}_{}'.format(np.argmax(view_psnr) // 9 + 1, np.argmax(view_psnr) % 9 + 1),
							  np.max(view_ssim),'{}_{}'.format(np.argmax(view_ssim) // 9 + 1, np.argmax(view_ssim) % 9 + 1),
							  np.mean(view_psnr), np.mean(view_ssim), np.min(view_psnr), np.min(view_ssim),lftime])
			set_psnr.append(np.mean(view_psnr))
			set_ssim.append(np.mean(view_ssim))
			if not path.exists('./{}/test_result'.format(result_dir)):
				makedirs('./{}/test_result'.format(result_dir))
			p = pandas.DataFrame(view_factor)
			p.to_csv('./{}/test_result/test_view_psnr_ssim.csv'.format(result_dir))
			p = pandas.DataFrame(lf_factor)
			p.to_csv('./{}/test_result/test_LF_view_psnr_ssim.csv'.format(result_dir))
			###计算epi的psnr和ssim
			lf_epi_factor.append([lf, np.max(epi_psnr), '{}_{}'.format(np.argmax(epi_psnr) // h + 1, np.argmax(epi_psnr) % h + 1),
				 np.max(epi_ssim), '{}_{}'.format(np.argmax(epi_ssim) // h + 1, np.argmax(epi_ssim) % h + 1),
				 np.mean(epi_psnr), np.mean(epi_ssim), np.min(epi_psnr), np.min(epi_ssim)])
			set_epi_psnr.append(np.mean(epi_psnr))
			set_epi_ssim.append(np.mean(epi_ssim))
			p = pandas.DataFrame(epi_factor)
			p.to_csv('./{}/test_result/test_epi_psnr_ssim.csv'.format(result_dir))
			p = pandas.DataFrame(lf_epi_factor)
			p.to_csv('./{}/test_result/test_LF_epi_psnr_ssim.csv'.format(result_dir))
	###计算数据集VIEW的PSNR和SSIM
	set_factor.append([set,np.mean(set_psnr), np.mean(set_ssim)])
	p = pandas.DataFrame(set_factor)
	p.to_csv('./{}/test_result/test_set_view_psnr_ssim.csv'.format(result_dir))
	###计算数据集VIEW的PSNR和SSIM、
	set_epi_factor.append([set, np.mean(set_epi_psnr), np.mean(set_epi_ssim)])
	p = pandas.DataFrame(set_epi_factor)
	p.to_csv('./{}/test_result/test_set_epi_psnr_ssim.csv'.format(result_dir))
###计算所有数据集的PSNR和SSIM
set_factor.append(['all',np.mean([x[5] for x in lf_factor[1:]]),np.mean([x[6] for x in lf_factor[1:]])])
p = pandas.DataFrame(set_factor)
p.to_csv('./{}/test_result/test_set_view_psnr_ssim.csv'.format(result_dir))
###计算所有数据集epi的PSNR和SSIM
set_epi_factor.append(['all', np.mean([x[5] for x in lf_epi_factor[1:]]), np.mean([x[6] for x in lf_epi_factor[1:]])])
p = pandas.DataFrame(set_epi_factor)
p.to_csv('./{}/test_result/test_set_epi_psnr_ssim.csv'.format(result_dir))

####预测结果y转rgb
def process(y,cb,cr,a):
	ycbcr_img=np.ones(a)
	cb_n=cv2.resize(cb,(a[1]//2,a[0]//2),interpolation=cv2.INTER_CUBIC)
	cb_n = cv2.resize(cb_n, (a[1], a[0]), interpolation=cv2.INTER_CUBIC)
	cr_n = cv2.resize(cr, (a[1] // 2, a[0] // 2), interpolation=cv2.INTER_CUBIC)
	cr_n = cv2.resize(cr_n, (a[1], a[0]), interpolation=cv2.INTER_CUBIC)
	cr_n=np.clip(cr_n,16/255,235/255)
	cb_n = np.clip(cb_n, 16 / 255, 235 / 255)
	ycbcr_img[:, :, 0] = (y*255)
	ycbcr_img[:, :, 1] = (cb_n*255)
	ycbcr_img[:, :, 2] = (cr_n*255)
	rgb_img=color.ycbcr2rgb(ycbcr_img)
	rgb_img = np.clip(rgb_img,0,1)
	rgb_img=np.uint8(np.round(rgb_img*255))
	return rgb_img
