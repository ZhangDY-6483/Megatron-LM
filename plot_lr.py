#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import re

def smooth(x, window_len=5, window='flat'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

def compare_strategy(job_file1, job_file2, prefix=", loss: (\S+),", fig_name='', window_len=5, skip_step=50):
    #re_str = prefix
    re_str = "learning rate: (\S+)e\-(\S+)"
    index = 0
    a_value = []
    with open(job_file1, "r") as f:    #打开文件
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            if 'valid' in line:
                continue
            s = re.search(re_str, line)  # .group(1) 
            if s:
                if index >= skip_step:
                    a_value.append(float(s.group(1)))
                else:
                    index += 1

    f.close()
    
    index = 0
    re_str = "learning rate: (\S+)E\-(\S+)"
    b_value = []
    with open(job_file2, "r") as f:    #打开文件
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            if 'valid' in line:
                continue
            s = re.search(re_str, line)  # .group(1) 
            if s:
                if index >= skip_step:
                    b_value.append(float(s.group(1)))
                else:
                    index += 1
    f.close()

    # print(b_value)
    b_value = b_value[:]
    min_len = min(len(a_value), len(b_value))

    plt.figure(1, figsize=(12, 6))
    plt.subplot(1, 2, 1) #图一包含1行2列子图，当前画在第一行第一列图上
    plt.plot(smooth(np.asarray(a_value[:min_len])), color='red', label=job_file1)
    plt.plot(smooth(np.asarray(b_value[:min_len])), color='green', label=job_file2)

    leg = plt.legend()
    leg.get_frame().set_linewidth(0.0)

    plt.xlabel('steps')
    plt.ylabel('lr')


    plt.figure(1, figsize=(12, 6))
    plt.subplot(1, 2, 2) #图一包含1行2列子图，当前画在第一行第一列图上
    diff = [b_value[i] - a_value[i] for i in range(min_len)]
    plt.plot(smooth(np.asarray(diff[:min_len])), color='blue', label="diff")
    plt.xlabel('steps')
    plt.ylabel('diff')

    # plt.show()
    plt.savefig(fig_name)


# compare_strategy("./log_amp_test_C1/workerlog.0", "./log_amp_base_C1/workerlog.0", fig_name='./figure_loss_C1_fp16_test_vs_base.jpg', window_len=5, skip_step=50)

#compare_strategy("fleetx_log/fleetx_fp32_10000steps.log", "log/345_single_fp32.log", fig_name='./figure_lr_fp32.jpg', window_len=5, skip_step=0)
#compare_strategy("fleetx_log/fleetx_345M_single_fp32.log", "log/345_single_fp32.log", fig_name='./figure_lr_fp32.jpg', window_len=5, skip_step=0)
compare_strategy("fleetx_log/single_345M_fp32", "log/345_single_fp32.log", fig_name='./figure_loss_fp32_lr.jpg', window_len=5, skip_step=0)

