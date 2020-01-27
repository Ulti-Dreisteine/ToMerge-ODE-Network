# -*- coding: utf-8 -*-
"""
Created on 2019/11/22 上午9:25

@Project -> File: ode-neural-network -> __init__.py.py

@Author: luolei

@Describe: 初始化
"""

from mod.config.config_loader import config

# 设备和动力学参数
V = config.conf['V']
k_0 = config.conf['k_0']
E_div_R = config.conf['E_div_R']
ro = config.conf['ro']
roc = config.conf['roc']
Cp = config.conf['Cp']
Cpc = config.conf['Cpc']
qc = config.conf['qc']
hA = config.conf['hA']
delta_H = config.conf['delta_H']
Tc_0 = config.conf['Tc_0']

__all__ = [
	'V', 'k_0', 'E_div_R', 'ro', 'roc', 'Cp', 'Cpc', 'qc', 'hA', 'delta_H', 'Tc_0'
]