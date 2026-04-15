#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : KittenSleepwalk
# @time    : 2025/1/14 下午4:10
# @function: can merge data by this file or use run.py while training
# @version : V1.0.0
import Integration.privateDataset.Merge as Merge

if __name__ == '__main__':
    print('+++++++++++++Test merge process')
    Merge.MergeByYaml().work()
    print('+++++++++++++Test load failure')
    Merge.MergeByYaml().not_work(0)
    print('+++++++++++++Test load success')
    Merge.MergeByYaml().not_work(1)
