import torch
import torch.nn as nn
import torch.nn.init as init

def CR(inp, oup,k):
	return nn.Sequential(
		nn.Conv2d(inp, oup, kernel_size=k, stride=1, padding=(k-1)//2),
		nn.ReLU())
def CS(inp, oup):
	return nn.Sequential(
		nn.Conv2d(inp, inp, kernel_size=3, stride=1, padding=1),
		nn.Sigmoid())
def res(inp, oup):
	return nn.Sequential(
		nn.Conv2d(inp, oup, kernel_size=3, stride=1, padding=1),
		nn.ReLU(),
		nn.Conv2d(oup, inp, kernel_size=3, stride=1, padding=1))

class Net(nn.Module):
	def __init__(self, rand_num,spa_upscale,ang_upscale):
		super(Net, self).__init__()
		self.cr0 = CR(2, 8*ang_upscale,3)
		self.cr2 = CR(2, 8*ang_upscale,3)
		self.cr4 = CR(2, 4*ang_upscale,3)
		self.cr6 = CR(1, 8*ang_upscale,3)
		self.cr8 = CR((8*2*2+4*2+8*4)*ang_upscale,32*ang_upscale,3)

		self.cr11 = CR(rand_num, 32, 3)
		self.cr13 = CR(32+32*ang_upscale, 64, 3)
		self.res1=res(32*ang_upscale,32*ang_upscale)
		self.res2 = res(32, 32)
		self.cs1 = CS(32*ang_upscale,32*ang_upscale)
		self.cs2 = CS(1,1)
		self.cs3 = CS(32, 32)
		self.relu = nn.ReLU()
		self.convf = nn.ConvTranspose2d(64, 32, spa_upscale+2*spa_upscale//2, spa_upscale, spa_upscale//2)
		self.convf2 = nn.ConvTranspose2d(4, (2+ang_upscale-1)**2, spa_upscale+2*spa_upscale//2, spa_upscale, spa_upscale//2)
		self.conv5 = nn.Conv2d(32, (2+ang_upscale-1)**2, 3,1,1)

		self._init_parameters()

	def _init_parameters(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, x1, x2, x3, x4, x_r, sobelimg):
		x_all = torch.cat((x1, x2, x3, x4), 1)
		x_all = self.convf2(x_all)
		x_0 = self.cr0(torch.cat((x1, x2), 1))
		x_1 = self.cr0(torch.cat((x3, x4), 1))
		x_2 = self.cr2(torch.cat((x1, x3), 1))
		x_3 = self.cr2(torch.cat((x2, x4), 1))
		x_4 = self.cr4(torch.cat((x1, x4), 1))
		x_5 = self.cr4(torch.cat((x2, x3), 1))
		x1 = self.cr6(x1)
		x2 = self.cr6(x2)
		x3 = self.cr6(x3)
		x4 = self.cr6(x4)
		x_6 = torch.cat((x1, x_0, x2, x_2, x_4, x_5, x_3, x3, x_1, x4), 1)
		x_6 = self.cr8(x_6)
		x_6 = self.res1(x_6)+x_6
		x_6=(self.cs1(x_6))*x_6
		x_com=[]
		for i in range(len(x_r)):
			s1=self.cs2(sobelimg[i])
			xr1=s1*x_r[i]
			x_com.append(xr1)
		x_r5 = torch.cat((x_com), 1)
		x_r5 = self.cr11(x_r5)
		x_r5 = self.res2(x_r5)+x_r5
		x_r5=self.cs3(x_r5)*x_r5
		x_8 = torch.cat((x_6, x_r5), 1)
		x_8 = self.cr13(x_8)
		x_f = self.convf(x_8)
		out = self.conv5(x_f)+x_all

		return out
