import msvcrt
import os
import sys

import readchar
import json

import numpy as np


def control_table(options: list):
    # options = ["1. 我配置文件忘改了，请终止程序; ", "2. 之前的文件不需要了，请删除重新生成; ",
    #            "3. 就使用这个数据集，跳过生成阶段; ", "4. 配置文件已经重新修改，请重新读取并生成; ",
    #            "5. 不想修改配置，备份一下重新开始生成."]
    selected_index = 0
    # 1. 初始打印菜单
    draw_menu(options, selected_index)
    while True:
        # 2. 获取按键 (非阻塞或单字符读取)
        key = readchar.readkey()

        # 逻辑处理
        if key == 'UP':
            selected_index = (selected_index - 1) % len(options)
        elif key == 'DOWN':
            selected_index = (selected_index + 1) % len(options)
        elif key == 'ENTER':
            print(f"\n>> 确认选择: {options[selected_index]}")  # 这里会换行，保留在历史记录里
            break
        # 3. 核心刷新步骤
        # 第一步：光标回到菜单起始位置 (上移 len(options) 行)
        move_cursor_up(len(options))

        # 第二步：重新绘制菜单 (覆盖旧内容)
        draw_menu(options, selected_index)


def draw_menu(options, selected_index):
    print("请使用上下键选择，回车确认：")
    for i, option in enumerate(options):
        # 简单的样式：选中项前面加星号
        prefix = "★ " if i == selected_index else "  "
        # 打印整行，注意行尾加空格防止旧字符残留
        print(f"{prefix}{option}          ")


def clear_line():
    """清除当前光标所在行的剩余内容"""
    sys.stdout.write("\033[K")


def move_cursor_up(lines):
    """光标上移指定行数"""
    sys.stdout.write(f"\033[{lines}A")


def get_key():
    """获取按键，返回方向键的代号或字符"""
    if os.name == 'nt':  # Windows
        key = msvcrt.getch()
        # 方向键在Windows下通常是两个字符，第一个是 b'\xe0' 或 b'\x00'
        if key == b'\xe0' or key == b'\x00':
            key2 = msvcrt.getch()  # 读取第二个字符来判断具体方向
            if key2 == b'H': return 'UP'
            if key2 == b'P': return 'DOWN'
            if key2 == b'K': return 'LEFT'
            if key2 == b'M': return 'RIGHT'
            return 'OTHER'
        else:
            return key.decode('utf-8', errors='ignore')
    else:
        # 简单的非Windows兼容（Linux/Mac下通常读取 \033[A 等序列）
        # 这里为了演示简洁，主要适配Windows，Linux需配合 curses 或 readchar
        return ''


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理NumPy数据类型"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)  # 将 numpy.int64 等转换为 Python int
        elif isinstance(obj, np.floating):
            return float(obj)  # 将 numpy.float32 等转换为 Python float
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # 将 numpy 数组转换为 Python 列表
        else:
            return super(NumpyEncoder, self).default(obj)
