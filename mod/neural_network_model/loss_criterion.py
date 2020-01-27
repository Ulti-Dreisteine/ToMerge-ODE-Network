# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

损失函数
"""
import torch
from torch import nn
import torch.nn.functional as f


def smape(y_true, y_pred):
	"""
	平均百分比误差
	:param y_true: torch.Tensor, 真实值
	:param y_pred: torch.Tensor, 预测值
	:return: torch.tensor, smape结果
	"""
	
	numerator = torch.abs(y_pred - y_true)
	denominator = torch.div(torch.add(torch.abs(y_pred), torch.abs(y_true)), 2.0)
	
	return torch.mean(torch.abs(torch.div(numerator, denominator)))


class FocalLoss(nn.Module):
	def __init__(self):
		super(FocalLoss, self).__init__()
	
	def forward(self, y_true, y_pred):
		error = torch.norm(y_true - y_pred, dim = 1)
		p = f.softmax(error, dim = 0)
		loss = torch.sum(torch.mul(p, error))
		return loss


def criterion(y_true, y_pred):
	"""
	损失函数, 目前的损失函数无法解决稳态数据较长时的模型学习
	:param y_true: torch.Tensor, 真实值
	:param y_pred: torch.Tensor, 预测值
	:return: torch.tensor, loss函数结果
	"""
	
	# l1 = nn.L1Loss()
	# l2 = nn.MSELoss()
	# loss = l1(y_true, y_pred) + l2(y_true, y_pred)
	
	focal_loss = FocalLoss()
	loss = focal_loss(y_true, y_pred)
	
	return loss


if __name__ == '__main__':
	import numpy as np
	import torch.nn.functional as f

	y_true = np.array(
		[
			[1, 1],
			[1, 1],
			[1, 1]
		]
	)

	y_pred = np.array(
		[
			[2, 2],
			[2, 2],
			[2, 3]
		]
	)

	y_true = torch.from_numpy(y_true.astype(np.float32))
	y_pred = torch.from_numpy(y_pred.astype(np.float32))
	
	# 计算loss
	error = torch.norm(y_true - y_pred, dim = 1)
	p = f.softmax(error, dim = 0)
	loss = torch.sum(torch.mul(p, error))
	
	
	