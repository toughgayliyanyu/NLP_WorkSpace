from tensorflow import keras
import tensorflow as tf
import tokenization
from run_classifier import InputFeatures, InputExample, DataProcessor, create_model, convert_examples_to_features
import numpy as np
# pb_file_path='./output_pbmodel/resaved_albert_tiny.pb'
pb_file_path='./output_pbmodel/resaved_albert_tiny_vectorall_sorted_1000.pb'

def get_sentence_examples(questions):
    examples = []
    guid = 'test-0'
    text_a = tokenization.convert_to_unicode(questions[0])
    label = str(0)
    examples.append(InputExample(guid=guid, text_a=text_a, label=label))
    print('examples',examples)
    return examples


predict_examples = get_sentence_examples([("基金")])
features = convert_examples_to_features(predict_examples, ['0', '1'], 32,
                                        tokenization.FullTokenizer(vocab_file='./albert_config/vocab.txt',
                                                                   do_lower_case=True))
input_ids = [features[0].input_ids]
print('input_ids:',input_ids)
input_mask = [features[0].input_mask]
print('input_mask:',input_mask)
segment_ids = [features[0].segment_ids]
print('segment_ids:',segment_ids)
#构造vectors_all输入向量
vecotr_in=np.load('test_data.npy')
vecotr_in=np.tile(vecotr_in,(1000,1)).reshape((1000,312)).astype(np.float32)
with tf.Session() as sess:
    #用上下文管理器打开pd文件
    with open(pb_file_path,"rb") as pd_flie:
        #获取图
        graph = tf.GraphDef()
        #获取参数
        graph.ParseFromString(pd_flie.read())
        #引入输入输出接口
        ins1,ins2,ins3,ins4,outs = tf.import_graph_def(graph,input_map={},return_elements=["input_ids:0","input_mask:0","segment_ids:0","vectors_all:0","concat:0"])
        # ins1, ins2, ins3, outs = tf.import_graph_def(graph, input_map={},return_elements=["input_ids:0", "input_mask:0","segment_ids:0","bert/pooler/dense/Tanh:0"])
        #进行预测
        print('output type:',outs.dtype)
        print('output shape:', outs.shape)
        print("y: ",sess.run(outs,{ins1:input_ids,ins2:input_mask,ins3:segment_ids,ins4:vecotr_in}))
        # print("y: ", sess.run(outs, {ins1: input_ids, ins2: input_mask, ins3: segment_ids}))
print('\n ===== predicting =====\n')