import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
import tokenization
from run_classifier import InputFeatures, InputExample, DataProcessor, create_model, convert_examples_to_features
import modeling
import json
import numpy as np

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
        self.vectors_all = tf.placeholder(dtype=tf.float32, shape=[1000, 312], name='vectors_all')

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
        x_norm = tf.nn.l2_normalize(output_layer, axis=-1)
        y_norm = tf.nn.l2_normalize(self.vectors_all, axis=-1)
        sim = tf.matmul(x_norm, tf.transpose(y_norm, [1, 0]))
        sort_predicted = tf.sort(sim, direction='DESCENDING')
        index = tf.argsort(sim, direction='DESCENDING')
        index = tf.cast(index, dtype=tf.float32)
        predicted_output = tf.concat([sort_predicted, index], axis=0)
        self.predictions = predicted_output

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
                     self.segment_ids: batch["segment_ids"],
                     self.vectors_all:batch["vecotr_in"]}
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

    def predict(self, input_id, input_mask, segment_id,vecotr_in):
        prediction = self.model.infer(self.sess,
                                      dict(input_ids=input_id,
                                           input_masks=input_mask,
                                           segment_ids=segment_id,
                                           vecotr_in=vecotr_in))
        return prediction

    def save_model(self,resaved_model_path):
        self.model.resaved(self.sess,resaved_model_path)


#####保存模型################
# Predictor().save_model('./resaved_albert_tiny_vectorall_sorted_1000/resaved_albert_tiny_vectorall_sorted_1000')



# ####创建输入实例################
vecotr_in=np.load('test_data.npy')
vecotr_in=np.tile(vecotr_in,(1000,1)).reshape((1000,312)).astype(np.float32)
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



predictor_0 = Predictor().predict(input_ids, input_mask, segment_ids,vecotr_in)
print('########基金#########')
print(predictor_0)
print(predictor_0.shape)

