import numpy as np

from Tools.draw import *

# x = [80,    90,     100, 110,   120,    130,   135,    137,    139,   140, 141,  150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 260]
# y = [0.172, 0.110, 0.119, 0.123, 0.126, 0.125, 0.200,  0.193, 0.227, 0.223, 0.261, 0.172, 0.156, 0.214, 0.121, 0.109, 0.100, 0.149, 0.219, 0.188, 0.176, 0.228]
x = [80, 90, 100, 110, 120, 125, 130, 135, 140, 141, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 260]
y = [0.172, 0.110, 0.119, 0.123, 0.126, 0.116, 0.125, 0.200, 0.223, 0.261, 0.172, 0.156, 0.214, 0.121, 0.109, 0.100,
     0.149, 0.219, 0.188, 0.176, 0.228]

draw_curve_and_special_star(x, y, 200, 'neuron num', 'error', 'S-MLP first hidden layer neuron num',
                            'IMG/S_MLP_h1_curve.png', split_x=10)
draw_line_and_special_star(x, y, 200, 'neuron num', 'error', 'S-MLP first hidden layer neuron num',
                           'IMG/S_MLP_h1_line.png', split_x=10)

x = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
y = [0.156, 0.110, 0.131, 0.109, 0.224, 0.142, 0.218, 0.204, 0.171, 0.155, 0.193, 0.190, 0.295, 0.178, 0.165, 0.100,
     0.146, 0.184, 0.123, 0.140, 0.191, 0.131, 0.278, 0.202, 0.207, 0.110, 0.132, 0.176]

draw_curve_and_special_star(x, y, 50, 'neuron num', 'error', 'S-MLP second hidden layer neuron num',
                            'IMG/S_MLP_h2_curve.png', split_x=1)
draw_line_and_special_star(x, y, 50, 'neuron num', 'error', 'S-MLP second hidden layer neuron num',
                           'IMG/S_MLP_h2_line.png', split_x=1)

x = [300, 400, 500, 600, 700, 800, 900, 950, 980, 1000, 1020, 1050, 1100, 1150, 1200, 1300, 1400, 1500, 1600]
y = [0.283, 0.128, 0.124, 0.108, 0.135, 0.238, 0.148, 0.130, 0.107, 0.100, 0.134, 0.166, 0.126, 0.113, 0.110, 0.130,
     0.171, 0.263, 0.343]

draw_curve_and_special_star(x, y, 1000, 'memory capacity', 'error', 'memory capacity of Memory-Recall layer',
                            'IMG/MR_memory_capacity_curve.png', split_x=50)
draw_line_and_special_star(x, y, 1000, 'memory capacity', 'error', 'memory capacity of Memory-Recall layer',
                           'IMG/MR_memory_capacity_line.png', split_x=50)

# save_path = 'memory_capacity.png'
# # save_path = 'memory_capacity_noAnti.png'
#
# labels = ['1k', '2k', '4k']
# labels_name = 'memory_capacity'
# values = [2.483, 2.485, 2.606]
# # values = [0.065, 0.065, 0.068]
# values_name = 'MAE error'
# title = 'Hyperparameter Comparisons'
#
# draw_bar(save_path, values, labels, values_name, labels_name, title)


# save_path = 'Hyperparameter in S-MLP.png'
#
# labels = ['200-30', '200-40', '200-50', '200-55', '200-60']
# labels_name = 'two hidden layers neurons'
# values = [[2.806, 2.728], [2.886, 2.921], [2.472, 2.786], [2.966, 2.868], [2.684, 2.854]]
# values_name = 'MAE error'
# bar_type_names = ['dropout during training', 'no dropout during training']
# title = 'Hyperparameter Comparisons'
#
# values = np.array(values).T
# draw_multiBar_in1x(save_path, values, bar_type_names, labels, values_name, labels_name, title)


# save_path = 'Hyperparameter in T-MLP.png'
# # save_path = 'memory_capacity_noAnti.png'
#
# labels = ['1080-64', '1215_72', '1350-80', '1485-88','1620-96']
# labels_name = 'two hidden layers neurons'
# values = [3.113, 2.999, 2.472, 2.901, 2.781]
# values_name = 'MAE error'
# title = 'Hyperparameter Comparisons'
#
# values = np.array(values).T
# draw_bar(save_path, values, labels, values_name, labels_name, title)
