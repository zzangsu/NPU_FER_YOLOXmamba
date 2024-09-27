# -*- coding: utf-8 -*-
import datetime
import os
import maccel
import onnxruntime as ort

from utils.callbacks import EvalCallback
from utils.utils import get_classes


if __name__ == "__main__":

    seed            = 11
    classes_path    = 'model_data/rafdb_classes.txt'
    input_shape     = [320,320]
    save_dir            = 'logs/debug'
    eval_flag           = True
    eval_period         = 1
    num_workers         = 0
    onnx_test           = True
    onnx_path           = 'onnx_models/originalV2_320_finetune_woVSS_Unfreeze100.onnx'
    mxq_test            = False
    mxq_path            = 'mxq_models/originalV2_320_finetune_woVSS_Unfreeze100.mxq'
    model_type          = 'onnx'

    val_annotation_path     = 'model_data/rafdb_test_NPUpc.txt'

    if onnx_test:
        model_eval = ort.InferenceSession(onnx_path)
    if mxq_test:
        acc1 = maccel.Accelerator()
        
        mc1 =maccel.ModelConfig()
        mc1.exclude_all_cores()
        mc1.include(maccel.Cluster.Cluster1, maccel.Core.Core2)

        model_eval = maccel.Model(mxq_path, mc1)
        model_eval.launch(acc1)
        

    class_names, num_classes = get_classes(classes_path)


    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    
    
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    os.makedirs(log_dir)
    
    eval_callback   = EvalCallback(model_type,model_eval, input_shape, class_names, num_classes, val_lines, log_dir, \
                                        eval_flag=eval_flag, period=eval_period)
    print('Start Test')
    eval_callback.on_epoch_end(1, model_eval)
    print('Finish Validation')
        
