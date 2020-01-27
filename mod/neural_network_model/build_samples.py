# -*- coding: utf-8 -*-
"""
Created on 2019/12/2 下午5:00

@Project -> File: ode-net-model-research -> build_samples.py

@Author: luolei

@Describe:
"""

import torch
import numpy as np
import copy

from lib import vars, vars_n, discrete_t_steps
from lib import vstack_arr


# def vstack_arr(total, sub):
# 	"""竖向拼接数据"""
#
# 	if total is None:
# 		total = sub
# 	else:
# 		total = np.vstack((total, sub))
#
# 	return total


def get_integrate_t_arr(t0, dt, discrete_t_steps):
	"""
	获取各个点的积分时刻表
	:param t0: torch.tensor, 起始时间序列
	:param dt: torch.tensor, 时间间隔
	:param discrete_t_steps: int, 起始时间和结束时间离散个数
	:return: integrate_t_arr, torch.tensor, 积分各时刻表
	"""
	
	t, integrate_t_arr = copy.deepcopy(t0), copy.deepcopy(t0)
	for step in range(discrete_t_steps):
		t += dt
		integrate_t_arr = np.hstack((integrate_t_arr, t))
	
	return integrate_t_arr


def build_samples(total_obs_data):
	"""
	构造样本
	:param total_obs_data: pd.DataFrame, 所有观测样本表
	:return: x0, x1, dt, integrate_t_arr
	"""
	
	labels = list(total_obs_data['label'].drop_duplicates())
	
	x0, x1 = None, None
	dt = None
	integrate_t_arr = None
	
	for label in labels:
		sub_data = total_obs_data[total_obs_data.label == label].copy()
		
		sub_x0 = sub_data.iloc[: -1][vars].to_numpy()       # 获得初值x0
		sub_x1 = sub_data.iloc[1:][vars].to_numpy()         # 获得终值x1
		
		sub_t0 = sub_data.iloc[: -1][['time']].to_numpy()   # 获得初始时间t0
		sub_t1 = sub_data.iloc[1:][['time']].to_numpy()     # 获得终止时间t1
		
		sub_dt = (sub_t1 - sub_t0) / discrete_t_steps       # 获得迭代时间步长dt
		sub_integrate_t_arr = get_integrate_t_arr(sub_t0, sub_dt, discrete_t_steps)  # 获得积分时间步arr
		
		# 数据拼接
		x0 = vstack_arr(x0, sub_x0)
		x1 = vstack_arr(x1, sub_x1)
		dt = vstack_arr(dt, sub_dt)
		integrate_t_arr = vstack_arr(integrate_t_arr, sub_integrate_t_arr)
	
	# 转为tensor
	x0_tensor = torch.from_numpy(x0.astype(np.float32))
	x1_tensor = torch.from_numpy(x1.astype(np.float32))
	integrate_t_tensor = torch.from_numpy(integrate_t_arr.astype(np.float32))
	
	dt = torch.from_numpy(dt.astype(np.float32))
	dt_tensor = dt.mm(torch.ones(1, vars_n))  # 后续计算会用到
	
	return x0_tensor, x1_tensor, integrate_t_tensor, dt_tensor


