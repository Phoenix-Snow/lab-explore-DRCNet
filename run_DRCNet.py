import os
import time
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from Integration.addDataet.Params import Params

if __name__ == "__main__":
    param = Params()

    task = param.TASK
    shutdown_after_finish = False
    if task == 'train':
        from Exp.DRCNet.Train.DRCNet_RUN import Procedure
        from Tools.ChinaClimatDataset.Reader import StreamForecastingDataset, Register

        epoches_num = param.CONFIG['run']['epoches']
        batch_size = param.CONFIG['run']['batch_size']
        shutdown_after_finish = param.CONFIG['is_shutdown_after_finish']

        # # Register: Normalizer and AntiNormalizer
        # normalizer = Normalizer(param.RESOURCE['normal_params'], param.CONFIG['preprocess']['normalize_type'])
        # normalizer_input, normalizer_label, antiNormalizer = normalizer.normalization_input, normalizer.normalization_label, normalizer.antiNormalization
        # Register: Reader params & antiNormalization
        register_class_train = Register(deepcopy(param.RESOURCE['station_path_dict']), param.RESOURCE['train_list'],
                                        # param.TIME_SERIES_LENGTH, normalizer_input, normalizer_label,
                                        # [param.loc_column_date] + param.in_columns_1st + param.in_columns_2nd + param.in_columns_3rd,
                                        param.TIME_SERIES_LENGTH,
                                        deepcopy(param.RESOURCE['normal_params']),
                                        param.CONFIG['preprocess']['normalize_type'],
                                        param.in_columns_1st, param.in_columns_2nd + param.in_columns_3rd,
                                        param.out_column)
        register_class_val = Register(deepcopy(param.RESOURCE['station_path_dict']), param.RESOURCE['valid_list'],
                                      # param.TIME_SERIES_LENGTH, normalizer_input, normalizer_label,
                                      # [param.loc_column_date] + param.in_columns_1st + param.in_columns_2nd + param.in_columns_3rd,
                                      param.TIME_SERIES_LENGTH,
                                      deepcopy(param.RESOURCE['normal_params']),
                                      param.CONFIG['preprocess']['normalize_type'],
                                      param.in_columns_1st, param.in_columns_2nd + param.in_columns_3rd,
                                      param.out_column)
        antiNormalizer = register_class_train.antiNormalizer

        # dataset: train, valid
        Train_Dataset = StreamForecastingDataset(register_class_train, 'Train4CalcLoss')
        Val_Dataset = StreamForecastingDataset(register_class_val, 'Val4CalcLoss')
        Train_Dataset_Training = StreamForecastingDataset(register_class_train, 'Train4Forward')

        Train_Reader: DataLoader = DataLoader(Train_Dataset, batch_size=int(batch_size * 1), shuffle=False,
                                              num_workers=0)
        Train_Reader_Training: DataLoader = DataLoader(Train_Dataset_Training, batch_size=batch_size, shuffle=True,
                                                       num_workers=2)  # 只有训练需要打乱

        Valid_Reader: DataLoader = DataLoader(Val_Dataset, batch_size=int(batch_size * 1), shuffle=False,
                                              num_workers=0)

        # Model EXP (Register: Model, Loss Function, Optimizer, Trainer/Evaluator/Predictor Function)
        exp = Procedure(param.TIME_SERIES_LENGTH, param.CONFIG['run']['learning_rate'], 'cuda:0',
                        antiNormalizer, Train_Reader, Valid_Reader, param.CONFIG['run']['model_save_path'],
                        'supplement_' + param.NAME,
                        register_class_train.t_features_index, register_class_train.s_features_index,
                        register_class_train.label_features_index)
        exp.load_model(param.CONFIG['run']['pretrained_model_path_time_Module'],
                       param.CONFIG['run']['pretrained_model_path_total_Model'])  # 内部逻辑：有则加载, 无则pass

        total_steps = len(Train_Reader_Training)

        # Release 'param' Memory
        del param


        # EXP Forward Function
        def exp_forward(exp, forward_data, backward_target, epoch_num, batch_num):
            with torch.cuda.device(exp.__device__):
                # 训练
                loss = exp({'map': forward_data, 'label': backward_target})
                # 验证
                if (
                        (isinstance(loss, float) and bool(loss < float(exp.eval_loss_threshold))) or
                        (isinstance(loss, list) and isinstance(exp.eval_loss_threshold, list) and
                         any(train_loss < exp.eval_loss_threshold[idx] for idx, train_loss in enumerate(loss)))
                ):
                    exp.evaluate(epoch_num, batch_num)
                    evaluate = True
                else:
                    evaluate = False
            return loss, evaluate


        print()
        # EXP: Training
        # threadPoolCenter = ThreadPoolExecutor(max_workers=min(4, os.cpu_count()))  # 减少工作线程数量
        for epoch in range(epoches_num):
            start_time = time.time()

            batch_idx = -1
            for t, target in Train_Reader_Training:
                batch_idx += 1
                print(f'Epoch: {epoch + 1} / {epoches_num}, Step: {batch_idx + 1} / {total_steps}, ')

                # 并行执行
                # t1 = threadPoolCenter.submit(exp_forward, exp1, t, s_id, target)

                # 等待任务完成并获取结果
                # loss1 = t1.result()

                loss1 = exp_forward(exp, t, target, epoch, batch_idx + 1)

                print(f'Evaluate? : {loss1[1]}, Batch Loss: {loss1[0]}')

            epoch_time = time.time() - start_time
            print(
                f'                            @ Epoch: {epoch + 1} / {epoches_num} [cost time: {epoch_time:.2f}s], \n'
                f'                                  Min Loss: {exp.OUTPUT.min_loss}')

        if shutdown_after_finish:  # 如果确认需要训练结束后关闭电脑，需确保模型保存过
            print('即将执行关机程序，倒计时5分钟，如不需要请点击停止代码取消关机程序。。。')
            os.system('shutdown -s -t 300')  # 5分钟后执行关机

    elif task == 'test':
        from Exp.DRCNet.Train.DRCNet_TEST import Procedure
        from Tools.ChinaClimatDataset.Reader import StreamForecastingDataset, Register

        batch_size = param.CONFIG['run']['batch_size']
        shutdown_after_finish = param.CONFIG['is_shutdown_after_finish']

        # Register: Reader params & antiNormalization
        register_class_train = Register(param.RESOURCE['station_path_dict']['train'], None,
                                        # param.TIME_SERIES_LENGTH, normalizer_input, normalizer_label,
                                        # [param.loc_column_date] + param.in_columns_1st + param.in_columns_2nd + param.in_columns_3rd,
                                        param.TIME_SERIES_LENGTH,
                                        deepcopy(param.RESOURCE['normal_params']),
                                        param.CONFIG['preprocess']['normalize_type'],
                                        param.in_columns_1st, param.in_columns_2nd + param.in_columns_3rd,
                                        param.out_column)
        register_class_test = Register(param.RESOURCE['station_path_dict']['test'], None,
                                       # param.TIME_SERIES_LENGTH, normalizer_input, normalizer_label,
                                       # [param.loc_column_date] + param.in_columns_1st + param.in_columns_2nd + param.in_columns_3rd,
                                       param.TIME_SERIES_LENGTH,
                                       deepcopy(param.RESOURCE['normal_params']),
                                       param.CONFIG['preprocess']['normalize_type'],
                                       param.in_columns_1st, param.in_columns_2nd + param.in_columns_3rd,
                                       param.out_column)
        antiNormalizer = register_class_train.antiNormalizer

        # dataset: test
        Train_Dataset = StreamForecastingDataset(register_class_train, 'Train4CalcLoss')
        Test_Dataset = StreamForecastingDataset(register_class_test, 'TEST4CalcLoss')
        Train_Reader: DataLoader = DataLoader(Train_Dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        Test_Reader: DataLoader = DataLoader(Test_Dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Model EXP
        exp = Procedure(param.TIME_SERIES_LENGTH, param.CONFIG['run']['pretrained_model_path'], 'cuda:0',
                        antiNormalizer, Train_Reader, Test_Reader, param.CONFIG['run']['results_save_path'],
                        'supplement_' + param.NAME,
                        register_class_train.t_features_index, register_class_train.s_features_index,
                        register_class_train.label_features_index,
                        register_class_train.input_features_name, register_class_train.label_features_name,
                        is_test_train=param.CONFIG['run']['is_test_train'])
        exp.load_model(param.CONFIG['run']['pretrained_model_path'])

        total_steps = len(Test_Reader)

        # Release 'param' Memory
        del param
        print()
        exp()

    if shutdown_after_finish:  # 如果确认需要训练结束后关闭电脑，需确保模型保存过
        print('即将执行关机程序，倒计时5分钟，如不需要请点击停止代码取消关机程序。。。')
        os.system('shutdown -s -t 300')  # 5分钟后执行关机
