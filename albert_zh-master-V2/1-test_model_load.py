import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
import tokenization
from run_classifier import InputFeatures, InputExample, DataProcessor, create_model, convert_examples_to_features
import modeling
import json
import numpy as np
# init = tf.global_variables_initializer()
def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

class AlbertSentencePair(object):
    def __init__(self, is_training=True):
        self.__bert_config_path = "./albert_config/albert_config_tiny.json"
        self.__is_training = is_training

        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[1, 32], name='input_ids')
        self.input_masks = tf.placeholder(dtype=tf.int32, shape=[1, 32], name='input_mask')
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[1, 32], name='segment_ids')

        self.built_model()
        self.init_saver()

    def built_model(self):
        bert_config = modeling.BertConfig.from_json_file(self.__bert_config_path)

        model = modeling.BertModel(config=bert_config,
                                   is_training=False,
                                   input_ids=self.input_ids,
                                   input_mask=self.input_masks,
                                   token_type_ids=self.segment_ids,
                                   use_one_hot_embeddings=True)
        output_layer = model.get_pooled_output()
        self.predictions = output_layer

    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())

    def resaved(self,sess,path):
        self.saver.save(sess=sess,save_path=path,write_meta_graph=True)

    def infer(self, sess, batch):
        """
        预测新数据
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 预测结果
        """
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"]}
        predict = sess.run(self.predictions, feed_dict=feed_dict)
        return predict

class Predictor(object):
    def __init__(self):
        self.model = None

        # 创建模型
        self.create_model()
        # 加载计算图
        self.load_graph()

    def load_graph(self):
        """
        加载计算图
        :return:
        """
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state('./albert_tiny_zh')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format('./albert_tiny_zh'))

    def create_model(self):
        """
                根据config文件选择对应的模型，并初始化
                :return:
                """
        self.model = AlbertSentencePair(is_training=False)

    def predict(self, input_id, input_mask, segment_id):
        prediction = self.model.infer(self.sess,
                                      dict(input_ids=input_id,
                                           input_masks=input_mask,
                                           segment_ids=segment_id))
        return prediction

    def save_model(self,resaved_model_path):
        self.model.resaved(self.sess,resaved_model_path)


#####保存模型################
# Predictor().save_model('./resaved_albert_tiny/resaved_albert_tiny')



####创建输入实例进行预测################
def get_sentence_examples(questions):
    examples = []
    # for index, data in enumerate(questions):
    guid = 'test-0'
    text_a = tokenization.convert_to_unicode(questions[0])
            # text_b = tokenization.convert_to_unicode(str(data[1]))
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
segment_ids = [features[0].segment_ids]



predictor_0 = Predictor().predict(input_ids, input_mask, segment_ids)
print('########基金#########')
print(predictor_0)
print(predictor_0.shape)
print(predictor_0.dtype)



data_array=np.array([[]])
result_list=[]
for i in ['基金','基金经理','打篮球','理财产品','工商银行']:
    predict_examples = get_sentence_examples([(i)])
    features = convert_examples_to_features(predict_examples, ['0', '1'], 32,
                                                    tokenization.FullTokenizer(vocab_file='./albert_config/vocab.txt', do_lower_case=True))
    input_ids = [features[0].input_ids]
    input_mask = [features[0].input_mask]
    segment_ids = [features[0].segment_ids]
    # print('input_ids:',input_ids)
    #执行多次预测的时候每次要重置计算图
    tf.reset_default_graph()
    predictor = Predictor().predict(input_ids,input_mask,segment_ids)
    print('#################')
    # print(predictor)
    # predictor_0=np.r_[predictor_0,predictor]
    print(predictor.shape)
    sim_value=get_cos_similar_matrix(predictor_0,predictor)
    print('基金和'+i+'相似度：',sim_value[0][0])
    result_list.append(sim_value[0][0])
    print(sorted(enumerate(result_list),key=lambda x:x[1],reverse=True))
#
# np.save('data_array_5.npy',predictor_0)