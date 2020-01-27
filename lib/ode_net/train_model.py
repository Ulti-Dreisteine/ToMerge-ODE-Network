# -*- coding: utf-8 -*-
"""
Created on 2019/12/2 下午4:50

@Project -> File: ode-net-model-research -> train_model.py

@Author: luolei

@Describe: 训练模型
"""

import pandas as pd
import torch
import json
import sys
import logging

logging.basicConfig(level = logging.INFO)

import matplotlib.pyplot as plt

sys.path.append('../..')

from lib import lr, epochs
from lib import normalize
from mod.neural_network_model.build_samples import build_samples
from mod.neural_network_model.nn import PartialDeriveNet, ODESolver, integrate
from mod.neural_network_model.loss_criterion import criterion

#%% 是否使用GPU
use_cuda = torch.cuda.is_available()


def save_models(pd_net, train_loss_record):
	"""
	保存模型文件
	:param pd_net: torch.nn, torch训练的模型
	:param train_loss_record: list, 训练过程loss记录
	"""
	
	# 保存模型文件
	torch.save(pd_net.state_dict(), '../../file/model/state_dict.pth')
	
	# 保存模型结构参数
	model_struc_params = {
		'pd_net': {
			'input_size': pd_net.input_size,
			'hidden_sizes': pd_net.hidden_sizes,
			'output_size': pd_net.output_size
		}
	}
	
	with open('../../file/model/struc_params.json', 'w') as f:
		json.dump(model_struc_params, f)
	
	# 保存损失函数记录
	train_loss_list = [float(p.detach().cpu().numpy()) for p in train_loss_record]
	
	with open('../../file/model/train_loss.json', 'w') as f:
		json.dump(train_loss_list, f)


if __name__ == '__main__':
	# %% 载入观测数据并进行列归一化
	total_obs_data = pd.read_csv('../../data/runtime/total_obs_data.csv')
	total_obs_data = normalize(total_obs_data)
	
	# %% 构造样本
	x0_tensor, x1_tensor, integrate_t_tensor, dt_tensor = build_samples(total_obs_data)
	
	# %% 搭建网络
	input_size = x0_tensor.shape[1] + 1     # 输入为(x, t), 所以shape = dim(x) + dim(t)
	output_size = x0_tensor.shape[1]        # 输出为变量x的导数dx/dt, 所以shape = dim(x)
	hidden_sizes = [4 * input_size, 4 * input_size, 4 * output_size, 4 * output_size]
	
	pd_net = PartialDeriveNet(input_size, hidden_sizes, output_size)
	
	# %% 指定优化器
	optimizer = torch.optim.Adam(
		pd_net.parameters(),
		lr = lr
	)
	
	# %% 验证集设置
	x0_verify = x0_tensor[0: 1, :]
	t0_verify = total_obs_data[total_obs_data.label == 0]['time'].min()
	t1_verify = total_obs_data[total_obs_data.label == 0]['time'].max()
	dt_verify = 0.01
	
	# %% cuda
	if use_cuda:
		torch.cuda.empty_cache()
		x0_tensor = x0_tensor.cuda()
		x1_tensor = x1_tensor.cuda()
		integrate_t_tensor = integrate_t_tensor.cuda()
		dt_tensor = dt_tensor.cuda()
		pd_net = pd_net.cuda()
	
	train_loss_record = []
	plt.figure(figsize = [12, 5])
	for epoch in range(epochs):
		
		pd_net.train()
		
		if use_cuda:
			pd_net.to('cuda')
			
		x1_pred, _ = ODESolver(x0_tensor, integrate_t_tensor, dt_tensor, pd_net)  # TODO: 模型应该一直在GPU上，不要反复切换
		train_loss = criterion(x1_tensor, x1_pred)

		optimizer.zero_grad()
		train_loss.backward()
		optimizer.step()

		train_loss_record.append(train_loss)
		
		if (epoch + 1) % 50 == 0:
			print(epoch, train_loss)
		
		if (epoch + 1) % 200 == 0:
			pd_net.eval()
			_, x_verify_records = integrate(x0_verify, t0_verify, t1_verify, dt_verify, pd_net.to('cpu'))  # TODO: 模型应该一直在GPU上，不要反复切换

			x1_true_epoch = x1_tensor.cpu().numpy()
			x1_pred_epoch = x1_pred.detach().cpu().numpy()

			plt.clf()
			plt.subplot(1, 2, 1)
			plt.title('phase portrait')
			plt.scatter(x1_true_epoch[:, 0], x1_true_epoch[:, 1], c = '#1f77b4', s = 6, label = 'true')
			plt.scatter(x1_pred_epoch[:, 0], x1_pred_epoch[:, 1], c = '#ff7f0e', s = 6, label = 'pred')
			plt.legend(loc = 'lower left')

			plt.subplot(1, 2, 2)
			x1_pred_epoch = x_verify_records.detach().cpu().numpy()
			plt.title('phase portrait')
			plt.plot(x1_pred_epoch[:, 0, 0], x1_pred_epoch[:, 0, 1], c = '#ff7f0e', linewidth = 1.0, label = 'pred')
			plt.scatter(x1_true_epoch[:, 0], x1_true_epoch[:, 1], c = '#1f77b4', s = 6, label = 'true')
			plt.legend(loc = 'lower left')
			plt.pause(1.0)

		# 保存模型
		if (epoch + 1) % 500 == 0:
			print('saving model...')
			plt.savefig('../../graph/train_effect.png', dpi = 450)
			save_models(pd_net, train_loss_record)

			# break

	save_models(pd_net, train_loss_record)