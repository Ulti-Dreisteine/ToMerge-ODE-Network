# -*- coding: utf-8 -*-
"""
Created on 2019/11/22 下午5:03

@Project -> File: ode-neural-network -> eval_ode_net.py

@Author: luolei

@Describe: ODE Net模型评估
"""

import pandas as pd
import numpy as np
import torch
import json
import logging

logging.basicConfig(level = logging.INFO)

import matplotlib.pyplot as plt

from lib import proj_cmap
from lib import ca_0, T_0, q
from lib import steps, dt
from lib import vars_n, var_bounds
from lib.cstr_simulation.run import draw_system_phase_portrait
from mod.neural_network_model.nn import PartialDeriveNet, integrate


def load_models():
	"""
	载入已经训练好的模型
	:return: model: torch.nn, 训练好的pytorch模型
	"""
	
	with open('../../file/model/struc_params.json', 'r') as f:
		model_struc_params = json.load(f)
	
	model_path = '../../file/model/state_dict.pth'
	pretrained_model_dict = torch.load(model_path, map_location = 'cpu')
	
	input_size = model_struc_params['pd_net']['input_size']
	hidden_sizes = model_struc_params['pd_net']['hidden_sizes']
	output_size = model_struc_params['pd_net']['output_size']
	model = PartialDeriveNet(input_size, hidden_sizes, output_size)
	model.load_state_dict(pretrained_model_dict, strict = False)
	
	model.eval()
	
	return model


if __name__ == '__main__':
	# %% 载入数据
	total_obs_data = pd.read_csv('../../data/runtime/total_obs_data.csv')
	
	# %% 载入模型文件
	pd_net = load_models()
	
	# %% 参数
	ca_range, T_range = var_bounds['ca'], var_bounds['T']
	random_init_states_n = 100
	op_params = [ca_0, T_0, q]  # 操作参数
	t = np.arange(0, steps * dt, dt)
	
	t0 = 0.0
	t1 = steps * dt
	
	# %% 画实际相轨迹图
	draw_system_phase_portrait(ca_range, T_range, random_init_states_n, t, op_params)
	plt.scatter(total_obs_data.loc[:, 'ca'], total_obs_data.loc[:, 'T'], s = 3, c = proj_cmap['blue'])
	
	# %% 画预测相轨迹图
	ca_norm_range, T_norm_range = [0.0, 1.0], [0.0, 1.0]
	
	for i in range(random_init_states_n):
		ca, T = np.random.uniform(ca_norm_range[0], ca_norm_range[1]), np.random.uniform(T_norm_range[0], T_norm_range[1])
		x0 = torch.tensor([ca, T]).reshape(1, -1)
		_, x1_records = integrate(x0, t0, t1, dt, pd_net)
		
		x1_pred_epoch = x1_records.detach().cpu().numpy().reshape(-1, vars_n)
		
		# 还原值
		x1_pred_epoch[:, 0] = x1_pred_epoch[:, 0] * (ca_range[1] - ca_range[0]) + ca_range[0]
		x1_pred_epoch[:, 1] = x1_pred_epoch[:, 1] * (T_range[1] - T_range[0]) + T_range[0]
		
		plt.plot(x1_pred_epoch[:, 0], x1_pred_epoch[:, 1], '-', c = proj_cmap['orange'], linewidth = 0.6, alpha = 0.4)
	
	plt.xlim(ca_range)
	plt.ylim(T_range)
	plt.savefig('../../graph/eval_effect.png', dpi = 450)


