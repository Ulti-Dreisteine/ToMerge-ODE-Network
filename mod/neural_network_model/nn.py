# -*- coding: utf-8 -*-
"""
Created on 2019/11/22 上午10:22

@Project -> File: ode-neural-network -> nn.py

@Author: luolei

@Describe:
"""

import torch
import copy
from torch import nn
from torch.nn import init
import numpy as np


class PartialDeriveNet(nn.Module):
	"""
	内层用于计算偏导的神经网络模型
	"""
	
	def __init__(self, input_size, hidden_sizes, output_size):
		super(PartialDeriveNet, self).__init__()
		self.input_size = input_size
		self.hidden_sizes = hidden_sizes
		self.output_size = output_size
		
		self.bn_in = nn.BatchNorm1d(self.input_size)
		
		self.fc_0 = nn.Linear(self.input_size, self.hidden_sizes[0])
		self._init_layer(self.fc_0)
		self.bn_0 = nn.BatchNorm1d(self.hidden_sizes[0])
		
		self.fcs = []
		self.bns = []
		for i in range(len(hidden_sizes) - 1):
			fc_i = nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1])
			setattr(self, 'fc_{}'.format(i + 1), fc_i)
			self._init_layer(fc_i)
			self.fcs.append(fc_i)
			bn_i = nn.BatchNorm1d(self.hidden_sizes[i + 1])
			self.bns.append(bn_i)
		
		self.fc_out = nn.Linear(self.hidden_sizes[-1], self.output_size)
		self.prelu = nn.PReLU()  # todo: 这里面的 n_parameters 参数意义没有搞清楚
		self._init_layer(self.fc_out)
		self.bn_out = nn.BatchNorm1d(self.output_size)
	
	def _init_layer(self, layer):
		init.normal_(layer.weight)  # 使用这种初始化方式能降低过拟合
		init.normal_(layer.bias)
	
	def forward(self, x):
		x = self.bn_in(x)
		x = self.fc_0(x)
		x = self.bn_0(x)
		x = torch.sigmoid(x)
		
		for i in range(len(self.fcs)):
			x = self.fcs[i](x)
			x = torch.sigmoid(x)
		
		x = self.fc_out(x)
		x = self.bn_out(x)
		x = self.prelu(x)
		
		return x


def ODESolver(x0_tensor, integrate_t_tensor, dt_tensor, pd_net):
	"""
	ODE积分求解器
	:param x0: torch.tensor, shape = (points, dim_x)
	:param integrate_t_arr: torch.tensor, 用于确定每次积分步起止时刻的时间表，shape = (points, discrete_t_steps)
	:param dt_tensor: torch.tensor, shape = (points, dim_x)
	:param pd_net: torch.nn object, partial derive network obtained
	:return: x: torch.tensor, final integrated x, shape = (points, dim_x)
	:return: x_records: torch.tensor, records of x, shape = (time, points, dim_x)
	"""
	
	x = copy.deepcopy(x0_tensor)  # 初值
	x_records = x.unsqueeze(0)
	for step in range(integrate_t_tensor.shape[1] - 1):  # **注意积分计算次数
		input = torch.cat((x, integrate_t_tensor[:, step: step + 1]), dim = 1)
		partial_derives = pd_net(input)
		
		delta_x = torch.mul(partial_derives, dt_tensor)  # **点乘
		x = torch.add(x, delta_x)
		x_records = torch.cat((x_records, x.unsqueeze(0)), dim = 0)
	
	return x, x_records


def integrate(x0, t0, t1, dt, pd_net):
	"""
	计算初值问题的解
	:param x0: torch.tensor, shape = (points, dim_x)
	:param t0: float, initial time
	:param t1: float, final time
	:param dt: float, time interval len for integration
	:param pd_net: torch.nn object, partial derive net obtained
	:return: x: torch.tensor, final integrated x, shape = (points, dim_x)
	:return: x_records: torch.tensor, records of x, shape = (time, points, dim_x)
	"""
	
	integrate_t_arr = np.arange(t0, t1, dt).reshape(1, -1)
	integrate_t_tensor = torch.from_numpy(integrate_t_arr.astype(np.float32))
	
	dt = np.array([dt]).reshape(-1, 1)
	dt = torch.from_numpy(dt.astype(np.float32))
	dt_tensor = dt.mm(torch.ones(1, 2))
	
	x, x_records = ODESolver(x0, integrate_t_tensor, dt_tensor, pd_net)
	
	return x, x_records



