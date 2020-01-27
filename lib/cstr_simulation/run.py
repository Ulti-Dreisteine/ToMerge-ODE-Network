# -*- coding: utf-8 -*-
"""
Created on 2019/11/22 上午9:28

@Project -> File: ode-neural-network -> run.py

@Author: luolei

@Describe: CSTR模型运行
"""

from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(0)

from lib import proj_cmap
from lib import ca_0, T_0, q
from lib import var_bounds
from lib import steps, dt, vars, obs_n, init_states_n
from mod.chemical_reaction.reaction_model import anisothermal_reaction


def draw_system_phase_portrait(ca_range, T_range, random_init_states_n, t, op_params):
	"""
	画出系统的相轨迹图
	:param ca_range: list, ca初值变化范围
	:param T_range: list, T初值变化范围
	:param random_init_states_n: int, 随机起始状态数
	:param t: np.array, 一维数组，积分时刻向量
	:param op_params: list like [ca_0, T_0, q], 操作参数向量
	"""
	
	plt.figure('phase portrait', figsize = [6, 6])
	for i in range(random_init_states_n):
		ca, T = np.random.uniform(ca_range[0], ca_range[1]), np.random.uniform(T_range[0], T_range[1])
		
		# %% 积分求解
		outputs, _ = odeint(anisothermal_reaction, (ca, T), t, (op_params,), full_output = True)
		outputs = outputs[: steps, :]
		
		# %% 画相轨迹图
		if outputs[-1, 1] < 380:  # 手动确定分辨吸引子的变量值
			c = proj_cmap['grey']
		else:
			c = proj_cmap['grey']
		
		plt.plot(outputs[:, 0], outputs[:, 1], '-', c = c, linewidth = 0.6)
	
	plt.xlabel('ca', fontsize = 10)
	plt.ylabel('T', fontsize = 10)
	plt.xlim(ca_range),
	plt.ylim(T_range)
	plt.xticks(fontsize = 6)
	plt.yticks(fontsize = 6)
	plt.title('Phase Portrait')
	plt.tight_layout()
	plt.show()
	

def get_samples(ca_list, T_list, t, op_params):
	"""
	获取样本
	:param ca_list: np.array, 一维数组，ca初值序列
	:param T_list: np.array, 一维数组，T初值序列
	:param t: np.array, 一维数组，积分时刻向量
	:param op_params: list like [ca_0, T_0, q], 操作参数向量
	:return: total_obs_data: pd.DataFrame, 总体观测数据表
	"""
	
	total_obs_data = None
	
	plt.figure('output time series', figsize = [6, 8])
	for i in range(init_states_n):
		ca, T = ca_list[i], T_list[i]
		full_data, _ = odeint(anisothermal_reaction, (ca, T), t, (op_params,), full_output = True)
		
		# 添加对应的时间记录
		full_data = np.hstack((t.reshape(-1, 1), full_data))
		
		# 取得观测点数据
		obs_locs = sorted(np.random.permutation(np.arange(steps)).tolist()[: obs_n])  # 按照先后顺序排序
		obs_data = full_data[obs_locs, :]
		
		for var in vars:
			idx = vars.index(var)
			plt.subplot(len(vars), 1, idx + 1)
			plt.scatter(obs_data[:, 0], obs_data[:, idx + 1], s = 10, c = proj_cmap['blue'])
			plt.plot(full_data[:, 0], full_data[:, idx + 1], '--', c = proj_cmap['grey'], linewidth = 0.6)
			plt.xlabel('t', fontsize = 10)
			plt.ylabel(var, fontsize = 10)
			plt.xticks(fontsize = 6)
			plt.yticks(fontsize = 6)
			plt.tight_layout()
		
		# 数据记录
		columns = ['time'] + vars
		obs_data = pd.DataFrame(obs_data, columns = columns)
		obs_data['label'] = obs_data.apply(lambda x: i, axis = 1)
		
		if total_obs_data is None:
			total_obs_data = obs_data
		else:
			total_obs_data = pd.concat([total_obs_data, obs_data], axis = 0)
	
	total_obs_data = total_obs_data.reset_index(drop = True)
	
	# 保存图片和数据
	plt.suptitle('Temporal Variation')
	plt.tight_layout()
	plt.subplots_adjust(top = 0.944)
	plt.show()
	plt.savefig('../../graph/temporal_variation.png')
	total_obs_data.to_csv('../../data/runtime/total_obs_data.csv', index = False)
	
	return total_obs_data
	

if __name__ == '__main__':
	# %% 参数
	op_params = [ca_0, T_0, q]  # 操作参数
	t = np.arange(0, steps * dt, dt)
	
	# %% 生成多初值积分候选随机数据
	ca_list = np.random.permutation(np.linspace(0.13, 0.3, init_states_n))
	T_list = np.random.permutation(np.linspace(410, 450, init_states_n))
	total_obs_data = get_samples(ca_list, T_list, t, op_params)

	# %% 画相轨迹图
	ca_range, T_range = var_bounds['ca'], var_bounds['T']
	random_init_states_n = 500

	draw_system_phase_portrait(ca_range, T_range, random_init_states_n, t, op_params)
	plt.scatter(total_obs_data.loc[:, 'ca'], total_obs_data.loc[:, 'T'], s = 5, c = proj_cmap['blue'])
	plt.savefig('../../graph/phase_portrait_for_ode_system.png')
	
	

	




