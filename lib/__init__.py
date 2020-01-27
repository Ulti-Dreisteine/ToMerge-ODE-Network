# -*- coding: utf-8 -*-
"""
Created on 2019/12/2 下午4:04

@Project -> File: ode-net-model-research -> __init__.py.py

@Author: luolei

@Describe:
"""

import numpy as np

from mod.config.config_loader import config

# 项目设置
proj_cmap = config.proj_cmap

# 设备操作参数
ca_0 = config.conf['ca_0']
T_0 = config.conf['T_0']
q = config.conf['q']

# ode模拟运行参数
dt = config.conf['dt']
steps = config.conf['steps']
obs_n = config.conf['obs_n']
init_states_n = config.conf['init_states_n']

# 变量参数
vars = config.conf['vars']
var_bounds = config.conf['var_bounds']

# 神经网络参数
discrete_t_steps = config.conf['discrete_t_steps']

lr = config.conf['lr']
epochs = config.conf['epochs']

vars_n = vars.__len__()


def normalize(data):
	"""
	各字段归一化
	:param data:
	:return:
	"""
	
	data = data.copy()
	for col in vars:
		bounds = var_bounds[col]
		data[col] = data[col].apply(lambda x: (x - bounds[0]) / (bounds[1] - bounds[0]))
	
	return data


def vstack_arr(total, sub):
	"""竖向拼接数据"""
	
	if total is None:
		total = sub
	else:
		total = np.vstack((total, sub))
	
	return total


