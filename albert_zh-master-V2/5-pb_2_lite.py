# -*- coding:utf-8 -*-
##python 1
import numpy as np
import tensorflow as tf
in_path = "./output_pbmodel/resaved_albert_tiny_output_no_onehot.pb"

def representative_dataset_gen():
    yield [
        np.array([[101, 1825, 7032, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(np.int32),
        np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(np.int32),
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(np.int32)
    ]
# 模型输入节点
# input_tensor_name = ["input_ids","input_mask","segment_ids","vectors_all"]
input_tensor_name = ["input_ids","input_mask","segment_ids"]
# input_tensor_shape = {"input_ids":[1,32],"input_mask":[1,32],"segment_ids":[1,32],"vectors_all":[974,313]}
input_tensor_shape = {"input_ids":[1,32],"input_mask":[1,32],"segment_ids":[1,32]}
# 模型输出节点
# classes_tensor_name = ['concat']
classes_tensor_name = ['bert/pooler/dense/Tanh']
converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path,
                                            input_tensor_name, classes_tensor_name,
                                            input_tensor_shape)

converter.allow_custom_ops=True
#########动态量化参数：tf.lite.Optimize.DEFAULT:仅将权重从浮点数静态量化为整数，其精度为8比特
converter.optimizations=[tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
# converter.target_ops=[tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
##############量化模型参数，将权重精度从32位降低到8位##########
# converter.post_training_quantize = True
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

open("./output_litemodel/resaved_albert_tiny_output_no_onehot_int8.tflite", "wb").write(tflite_model)