import time
import tensorflow as tf
import tokenization
from run_classifier import InputExample, convert_examples_to_features
import numpy as np
import json
total_time_start=time.time()
time_load_start=time.time()
interpreter = tf.lite.Interpreter(model_path="./output_litemodel/resaved_albert_tiny_vectorall_sorted_1000.tflite")
# interpreter = tf.lite.Interpreter(model_path="./output_litemodel/resaved_albert_tiny_vectorall_sorted_1000.tflite")
time_load_end=time.time()
print('load model spend:',time_load_end-time_load_start)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

def get_sentence_examples(questions):
    examples = []
    guid = 'test-0'
    text_a = tokenization.convert_to_unicode(questions[0])
    label = str(0)
    examples.append(InputExample(guid=guid, text_a=text_a, label=label))
    return examples

data_time_start=time.time()
predict_examples = get_sentence_examples([("基金")])
features = convert_examples_to_features(predict_examples, ['0', '1'], 32,
                                        tokenization.FullTokenizer(vocab_file='./albert_config/vocab.txt',
                                                                   do_lower_case=True))
input_ids = [features[0].input_ids]
input_mask = [features[0].input_mask]
segment_ids = [features[0].segment_ids]
print('input_ids:',input_ids)
print('input_mask:',input_mask)
print('segment_ids:',segment_ids)
data_time_end=time.time()
print('create data time spend:',data_time_end-data_time_start)
#########构建多维词库####
vecotr_in=np.load('test_data.npy')
vecotr_in=np.tile(vecotr_in,(1000,1)).reshape((1000,312)).astype(np.float32)
t1 = time.time()
interpreter.set_tensor(input_details[0]['index'], input_ids)
interpreter.set_tensor(input_details[1]['index'], input_mask)
interpreter.set_tensor(input_details[2]['index'], segment_ids)
interpreter.set_tensor(input_details[3]['index'], vecotr_in)

#############################
interpreter.invoke()
t2 = time.time()
predictor_0 = interpreter.get_tensor(output_details[0]['index'])
print('model predict time spend:',t2-t1)
print('total time spend:',t2-total_time_start)

#[‘基金','基金经理','打篮球','理财产品','工商银行']
# predictor_0 = predictor_0.astype(np.float16)
print(predictor_0)
# np.set_printoptions(precision=3)
# np.save('predictor_zhuanzhang.npy',predictor_0)
# np.savetxt('predictor_0_0_0.txt',predictor_0,fmt='%.7e')
print(type(predictor_0))
print(predictor_0.dtype)
# prediction = prediction[0]
print(predictor_0.shape)
# with open('./word_file/word_list_1000.json','r',encoding='utf8') as fword:
#     word_list=json.load(fword)
# for i in predictor_0[1][:10]:
#     print(word_list[int(i)])
# with open('to_ameng_index.txt','w',encoding='utf8') as index_file:
#     for i in predictor_0[1]:
#         i=str(int(i))
#         index_file.write(i+'\r\n')


# print(b)

# words_list=[]
# with open('words_list.txt','r',encoding='utf-8') as files:
#     for words in files.readlines():
#         print(words.split()[0])
#         words_list.append(words.split()[0])
#
# print(words_list)
# #
# # for i in ['我的账户','查询电子回单','理财产品','工商银行']:
# # dic_index=dict.fromkeys(words_list)
# for i in ['基金','基金经理','打篮球','理财产品','工商银行']:
#         predict_examples = get_sentence_examples([(i)])
#         features = convert_examples_to_features(predict_examples, ['0', '1'], 32,
#                                                         tokenization.FullTokenizer(vocab_file='./albert_config/vocab.txt', do_lower_case=True))
#         input_ids = [features[0].input_ids]
#         input_mask = [features[0].input_mask]
#         segment_ids = [features[0].segment_ids]
#         # print('input_ids:',input_ids)
#
#         interpreter.set_tensor(input_details[0]['index'], input_ids)
#         interpreter.set_tensor(input_details[1]['index'], input_mask)
#         interpreter.set_tensor(input_details[2]['index'], segment_ids)
#         interpreter.invoke()
#         prediction_1 = interpreter.get_tensor(output_details[0]['index'])
#         # prediction_1_float16=prediction_1.astype(np.float16)
#         # dic_index[i] = prediction_1_float32.tolist()[0]
#         # predictor_0 = np.r_[predictor_0, prediction_1_float16]
#         sim_value=get_cos_similar_matrix(predictor_0,prediction_1)
#         print('基金和'+i+'相似度：',sim_value)
# #
# np.save('data_array_50.npy',predictor_0)
# dic_json=json.dumps(predictor_0.tolist())
# dic_json=json.dumps(dic_index,ensure_ascii = False)
# with open('data_array_50_float16.json','w+',encoding='utf8') as file:
#     file.write(dic_json)