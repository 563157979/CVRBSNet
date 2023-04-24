from PIL import Image
import torch.utils.data as data
import torchvision
from os import listdir
from os.path import join, isdir
import numpy as np
import random
from skimage import color,io
import cv2
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import torch
import albumentations
from math import sqrt
import os
from albumentations import (Blur,Flip,ShiftScaleRotate,GridDistortion,ElasticTransform,HorizontalFlip,CenterCrop,
                            HueSaturationValue,Transpose,RandomBrightnessContrast,CLAHE,RandomCrop,Cutout,CoarseDropout,
                            CoarseDropout,Normalize,ToFloat,OneOf,Compose,Resize,RandomRain,RandomFog,Lambda
                            ,ChannelDropout,ISONoise,VerticalFlip,RandomGamma,RandomRotate90,GaussNoise,MotionBlur,
							MedianBlur,OpticalDistortion,RGBShift,ChannelShuffle)

class DatasetFromFolder(data.Dataset):
	def __init__(self,image_dir,LF_order,set,aug,vol,rand_num):
		super(DatasetFromFolder, self).__init__()
		self.num=[]
		self.augmentation = aug
		self.set=set
		self.vol=vol
		self.rand_num=rand_num
		lf=listdir(image_dir)
		lf.sort()
		self.name=lf[LF_order]
		views=listdir(join(image_dir,lf[LF_order]))
		views.sort()
		img = io.imread(join(image_dir, lf[LF_order], '1_1.png'))
		[w,h,c]=np.shape(img)
		self.spatial_factor=4
		self.angr=17
		w,h=w//self.spatial_factor//self.spatial_factor*self.spatial_factor*self.spatial_factor,h//self.spatial_factor//self.spatial_factor*self.spatial_factor*self.spatial_factor
		if set=='training':
			factor = self.spatial_factor
		if set=='test':
			factor = self.spatial_factor//self.spatial_factor
		self.img = np.zeros((self.angr, self.angr, w//factor, h//factor), dtype=np.float32)
		self.down = np.zeros((self.angr, self.angr, w//factor//self.spatial_factor, h//factor//self.spatial_factor), dtype=np.float32)
		self.edge = np.zeros_like(self.down)
		for i in range(self.angr):
			for j in range(self.angr):
				img = io.imread(join(image_dir, lf[LF_order], '{}_{}.png'.format(i + 1, j + 1)))[:w, :h, :]
				if set=='training':
					img = cv2.resize(img,(0,0),fx=1/factor,fy=1/factor,interpolation=cv2.INTER_CUBIC)
				img=(color.rgb2ycbcr((img/255).astype(np.float32))[:,:,0]/255).astype(np.float32)
				self.img[i, j, :, :] =img
				img=cv2.resize(img,(0,0),fx=1/self.spatial_factor,fy=1/self.spatial_factor,interpolation=cv2.INTER_CUBIC)
				self.down[i, j, :, :]=img
				self.edge[i,j,:,:]=mycanny(img)
		if set=='training':
			self.patchsize = 64
			if factor==2:
				mode=[self.spatial_factor*2,self.spatial_factor*4]
				p = (w//factor) // (self.patchsize//2) * (h//factor) // (self.patchsize//2) * 5
			if factor==4:
				mode=[self.spatial_factor**2]
				p = (w//factor) // (self.patchsize//factor) * (h//factor) // (self.patchsize//factor) * 1
			if p>vol:
				self.vol=p
			print(w,h,factor,self.patchsize,self.vol)
		if set=='test':
			mode=[self.spatial_factor]
		for ds in mode:
			for i in range(1,self.angr,ds):
				for j in range(1,self.angr,ds):
					self.num.append([i,j,ds])
	def __getitem__(self, index):
		comb_num=len(self.num)
		input=[[] for i in range(2)]
		input_random=[]
		random_list = []
		input_random_sobel = []
		if self.set == 'training' or self.set == 'valid':
			a=[i for i in range(0,self.angr,self.spatial_factor**2)]
		if self.set == 'test':
			a=[i for i in range(0,self.angr,self.spatial_factor)]
		t1=self.num[index % comb_num][0] - 1
		t2=self.num[index % comb_num][1] - 1
		t3=self.num[index % comb_num][2]
		order_list=[[t1,t2],[t1+t3,t2],[t1,t2+t3],[t1+t3,t2+t3]]
		img_array,img_down,img_edge=self.img,self.down,self.edge
		wl=np.shape(img_array[0,0,:,:])
		if self.set == 'training' or self.set == 'valid':
			y=random.randint(0,wl[0]-self.patchsize)
			x=random.randint(0,wl[1]-self.patchsize)
			y,x=y//4*4,x//4*4
			y_down,x_down=y//4,x//4
		if self.augmentation and self.set == 'training':
			img_array=img_array[:,:,y:y+self.patchsize,x:x+self.patchsize]
			img_down=img_down[:,:,y_down:y_down+self.patchsize//self.spatial_factor,x_down:x_down+self.patchsize//self.spatial_factor]
			img_edge=img_edge[:,:,y_down:y_down+self.patchsize//self.spatial_factor,x_down:x_down+self.patchsize//self.spatial_factor]
			img_array, img_down, img_edge = augment(img_array, img_down, img_edge, self.spatial_factor)
		i=0
		while len(random_list)!=self.rand_num:
			p = random.choice(a)
			q = random.choice(a)
			if [p,q] not in random_list:
				if self.spatial_factor==2:
					if len(random_list)==self.rand_num-1:
						if [True for i in random_list if i not in order_list]:
							pass
						else:
							if [p,q] in order_list:
								continue
				random_list.append([p,q])
				input_random.append(img_down[p, q, :, :].copy())
				input_random_sobel.append(img_edge[p, q, :, :].copy())
				input_random[i] = ToTensor()(input_random[i][:, :, np.newaxis])
				input_random_sobel[i] = ToTensor()(input_random_sobel[i][:, :, np.newaxis])
				i += 1
		if self.set=='training' or self.set=='valid':
			target=np.zeros((self.patchsize,self.patchsize,(self.spatial_factor+1)**2),dtype='float32')
		ds = self.num[index % comb_num][2]
		for i in range(self.spatial_factor+1):
			for j in range(self.spatial_factor+1):
				if self.set=='training' or self.set == 'valid':
					target[:, :, i * (self.spatial_factor+1) + j] = img_array[self.num[index % comb_num][0] - 1 + i * ds // self.spatial_factor,self.num[index % comb_num][1] - 1 + j * ds // self.spatial_factor, :, :].copy()
				if i%self.spatial_factor == 0 and j%self.spatial_factor == 0:
					input[i // self.spatial_factor].append(img_down[self.num[index % comb_num][0] - 1 + i * ds // self.spatial_factor,self.num[index % comb_num][1] - 1 + j * ds // self.spatial_factor, :, :].copy())
					input[i // self.spatial_factor][j // self.spatial_factor] = ToTensor()(input[i // self.spatial_factor][j // self.spatial_factor][:, :, np.newaxis])
		if self.set == 'training' or self.set == 'valid':
			target=ToTensor()(target)
		if self.set == 'test':
			target=torch.from_numpy(img_array[::ds//self.spatial_factor,::ds//self.spatial_factor,:,:].copy())

		return input[0][0],input[0][1],input[1][0],input[1][1],target,input_random,input_random_sobel

	def __len__(self):
		return len(self.num)*self.vol

def mycanny(img):
	img1 = (img * 255).astype(np.uint8)
	dst=cv2.Canny(img1,50,100)
	kernel = np.ones((5, 5), np.uint8)
	dst = cv2.dilate(dst, kernel, 1)
	dst = dst.astype(np.float32)
	return dst
def color_aug(p=1):
	return Compose([
		OneOf([
			MotionBlur(p=1),
			Blur(blur_limit=3, p=1),
		], p=1),
		OneOf([
			RandomBrightnessContrast(p=1),
			RandomGamma(p=1),
		], p=1),
	], p=p)

def augment(y_np,y_down,y_edge,factor):
	if factor==4:
		p=0.7
	if factor==2:
		p=0.5
	if random.random() < p:
		y_np = y_np[:, ::-1, :, ::-1]
		y_edge = y_edge[:, ::-1, :, ::-1]
		y_down = y_down[:, ::-1, :, ::-1]
	if random.random() < p:
		y_np = y_np[::-1, :, ::-1, :]
		y_edge = y_edge[::-1, :, ::-1, :]
		y_down = y_down[::-1, :, ::-1, :]
	if random.random() < p:
		y_np = y_np.transpose(1, 0, 3, 2)
		y_edge = y_edge.transpose(1, 0, 3, 2)
		y_down = y_down.transpose(1, 0, 3, 2)
	#rotate
	if random.random() < p:
		r_ang = np.random.randint(1, 4)
		y_np = np.rot90(y_np, r_ang, (2, 3))
		y_np = np.rot90(y_np, r_ang, (0, 1))
		y_edge = np.rot90(y_edge, r_ang, (2, 3))
		y_edge = np.rot90(y_edge, r_ang, (0, 1))
		y_down = np.rot90(y_down, r_ang, (2, 3))
		y_down = np.rot90(y_down, r_ang, (0, 1))
	if random.random() < 0.8:
		angh, angw, h, w = y_np.shape
		y_np = y_np.transpose(0, 2, 1, 3)
		y_np = y_np.reshape(-1, angw * w)
		y_np_aug = color_aug(1)(image=y_np)
		y_np = y_np_aug['image']
		y_np = y_np.reshape(angh, h, angw, w)
		y_np = y_np.transpose(0, 2, 1, 3)
		img=y_np.reshape(angh*angw,h,w)
		img=img.transpose(1,2,0)
		y_down= cv2.resize(img, (0, 0), fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_CUBIC)
		y_down=y_down.transpose(2,0,1)
		y_down=y_down.reshape(angh,angw,h//factor,w//factor)
	return y_np,y_down,y_edge
