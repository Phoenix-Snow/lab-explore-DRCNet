#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : KittenSleepwalk
# @time    : 2024/10/18 下午4:42
# @function: 
# @version : V1.0.0
from datetime import datetime


## 计时区间计算
def time_count(start_time_str, duration_limit):
    time_duration = (datetime.now() - datetime.strptime(start_time_str, '%Y%m%d%H%M%S')).total_seconds()
    if time_duration > duration_limit:
        print('Time is up! Stop training. ', str(int(time_duration)), ' s >= ',
              str(duration_limit), ' s')
        return 1
    else:
        print('time duration：{:02d}:{:02d}:{:02d}\n'.format(int(time_duration) // 3600,
                                                            int(time_duration) % 3600 // 60,
                                                            int(time_duration) % 60))
        return 0
