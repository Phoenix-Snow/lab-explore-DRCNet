import os
import time

import torch
from torch.utils.data import DataLoader

from Integration.Competition.Params import Params
from Tools.Normalizer_1D import Normalizer

if __name__ == "__main__":
    param = Params()

    task = param.TASK
    shutdown_after_finish = False
    if task == 'train':
        from Exp.Competition.Train.ST_MRNet_RUN import Procedure
        from Tools.XinJiangProgram.Reader import SimpleReaderTrain, build_dataset4now, SpatialReader, build_dataset4future

        epoches_num = param.CONFIG['run']['epoches']
        batch_size = param.CONFIG['run']['batch_size']
        shutdown_after_finish = param.CONFIG['is_shutdown_after_finish']

        # Register: Normalizer and AntiNormalizer
        normalizer = Normalizer(param.RESOURCE['normal_params'], param.CONFIG['preprocess']['normalize_type'])
        normalizer_input, normalizer_label, antiNormalizer = normalizer.normalization_input, normalizer.normalization_label, normalizer.antiNormalization

        # dataset: train, valid
        if param.CONFIG['run']['predict_target'] == 'now':
            train_dataset, valid_dataset = build_dataset4now(param.RESOURCE['station_path_dict'],
                                                             param.RESOURCE['train_index_list'],
                                                             param.RESOURCE['valid_index_list'],
                                                             param.TIME_SERIES_LENGTH)
            target_mode = 'now'
        elif param.CONFIG['run']['predict_target'] == 'future':
            train_dataset, valid_dataset = build_dataset4future(param.RESOURCE['station_path_dict'],
                                                                param.RESOURCE['train_index_list'],
                                                                param.RESOURCE['valid_index_list'],
                                                                param.TIME_SERIES_LENGTH, False)
            target_mode = 'future'
        else:
            raise NotImplementedError
        # Reader - DataLoader(asynchronous): Train, Valid, Test, Config: batch_size, is_shuffle, num_workers
        Train_Simple = SimpleReaderTrain(train_dataset.input_t, train_dataset.label, train_dataset.station_l,
                                         normalizer_input, normalizer_label)
        Train_Reader: DataLoader = DataLoader(Train_Simple, batch_size=batch_size, shuffle=True, num_workers=0)
        Train_Simple_Training = SimpleReaderTrain(train_dataset.input_t, train_dataset.label, train_dataset.station_l,
                                                  normalizer_input, normalizer_label)
        Train_Reader_Training: DataLoader = DataLoader(Train_Simple, batch_size=batch_size, shuffle=True,
                                                       num_workers=0)
        Valid_Simple = SimpleReaderTrain(valid_dataset.input_t, valid_dataset.label, valid_dataset.station_l,
                                         normalizer_input, normalizer_label)
        Valid_Reader: DataLoader = DataLoader(Valid_Simple, batch_size=batch_size, shuffle=False, num_workers=0)
        del train_dataset, valid_dataset

        Spatial_Reader = SpatialReader(param.RESOURCE['station_map_dict'])

        # Model EXP (Register: Model, Loss Function, Optimizer, Trainer/Evaluator/Predictor Function)
        exp = Procedure(1, param.CONFIG['run']['learning_rate'], 'cuda:0',
                        antiNormalizer, param.CONFIG['run']['is_output_antiNormal'],
                        Train_Reader, Valid_Reader, Spatial_Reader, param.CONFIG['run']['model_save_path'])
        exp.load_model(param.CONFIG['run']['pretrained_model_path_total_Model'])  # 内部逻辑：有则加载, 无则pass

        total_steps = len(Train_Reader_Training)

        # Release 'param' Memory
        del param


        # EXP Forward Function for Thread
        def exp_forward(exp, forward_data, spatial_id, backward_target, epoch_num, batch_num):
            with torch.cuda.device(exp.__device__):
                loss = exp({'map': forward_data, 'spatial': spatial_id, 'label': backward_target})
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

        for epoch in range(epoches_num):
            start_time = time.time()

            batch_idx = -1
            for t, s_id, target in Train_Reader_Training:
                batch_idx += 1
                print(f'Epoch: {epoch + 1} / {epoches_num}, Step: {batch_idx + 1} / {total_steps}, ')

                loss1 = exp_forward(exp, t, s_id, target, epoch, batch_idx + 1)

                print(f'Evaluate? : {loss1[1]}, Batch Loss: {loss1[0]}')

            epoch_time = time.time() - start_time
            print(
                f'                            @ Epoch: {epoch + 1} / {epoches_num} [cost time: {epoch_time:.2f}s], \n'
                f'                                  Min Loss: {exp.OUTPUT.min_loss}')

        if shutdown_after_finish:  # 如果确认需要训练结束后关闭电脑，需确保模型保存过
            print('即将执行关机程序，倒计时5分钟，如不需要请点击停止代码取消关机程序。。。')
            os.system('shutdown -s -t 300')  # 5分钟后执行关机

    elif task == 'test':
        from Exp.Competition.Train.ST_MRNet_TEST import Procedure
        from Tools.XinJiangProgram.Reader import SimpleReaderTrain, build_test_dataset4now, SpatialReader, \
            build_test_dataset4future

        batch_size = param.CONFIG['run']['batch_size']
        shutdown_after_finish = param.CONFIG['is_shutdown_after_finish']

        # Register: Normalizer and AntiNormalizer
        normalizer = Normalizer(param.RESOURCE['normal_params'], param.CONFIG['preprocess']['normalize_type'])
        normalizer_input, normalizer_label, antiNormalizer = normalizer.normalization_input, normalizer.normalization_label, normalizer.antiNormalization

        # dataset: test
        if param.CONFIG['run']['predict_target'] == 'now':
            test_dataset = build_test_dataset4now(param.RESOURCE['station_path_dict']['test'], param.TIME_SERIES_LENGTH)
            train_dataset = build_test_dataset4now(param.RESOURCE['station_path_dict']['train'],
                                                   param.TIME_SERIES_LENGTH)
            target_mode = 'now'
        elif param.CONFIG['run']['predict_target'] == 'future':
            test_dataset = build_test_dataset4future(param.RESOURCE['station_path_dict']['test'],
                                                     param.TIME_SERIES_LENGTH, False)
            train_dataset = build_test_dataset4future(param.RESOURCE['station_path_dict']['train'],
                                                      param.TIME_SERIES_LENGTH, False)
            target_mode = 'future'
        else:
            raise NotImplementedError
        # Reader - DataLoader(asynchronous): Train, Valid, Test, Config: batch_size, is_shuffle, num_workers
        Test_Simple = SimpleReaderTrain(test_dataset.input_t, test_dataset.label, test_dataset.station_l,
                                        normalizer_input, normalizer_label)
        Test_Reader: DataLoader = DataLoader(Test_Simple, batch_size=batch_size, shuffle=False, num_workers=0)
        Train_Simple = SimpleReaderTrain(train_dataset.input_t, train_dataset.label, train_dataset.station_l,
                                         normalizer_input, normalizer_label)
        Train_Reader: DataLoader = DataLoader(Train_Simple, batch_size=batch_size, shuffle=False, num_workers=0)
        del test_dataset

        Spatial_Reader = SpatialReader(param.RESOURCE['station_map_dict'])

        # Model EXP (Register: Model, Loss Function, Optimizer, Trainer/Evaluator/Predictor Function)
        exp = Procedure(1, 'cuda:0', antiNormalizer, Test_Reader, Train_Reader, Spatial_Reader,
                        param.CONFIG['run']['results_save_path'], target_mode, True)
        exp.load_model(param.CONFIG['run']['pretrained_model_path'])  # 内部逻辑：有则加载, 无则pass

        total_steps = len(Test_Reader)

        # Release 'param' Memory
        del param
        print()
        exp()

    if shutdown_after_finish:  # 如果确认需要训练结束后关闭电脑，需确保模型保存过
        print('即将执行关机程序，倒计时5分钟，如不需要请点击停止代码取消关机程序。。。')
        os.system('shutdown -s -t 300')  # 5分钟后执行关机
