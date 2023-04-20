import time
import tensorflow as tf
import tokenization
from run_classifier import InputExample, convert_examples_to_features
import numpy as np
import json
import csv
total_time_start=time.time()
time_load_start=time.time()
interpreter = tf.lite.Interpreter(model_path="./output_litemodel/resaved_albert_tiny.tflite")
time_load_end=time.time()
print('load model spend:',time_load_end-time_load_start)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

##########加载词库##############
with open('./word_new.csv','r',encoding='utf8') as wordfile:
    reader=csv.reader(wordfile)
    rows = [row[0] for row in reader]
def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

def get_sentence_examples(questions):
    examples = []
    # for index, data in enumerate(questions):
    guid = 'test-0'
    text_a = tokenization.convert_to_unicode(questions[0])
            # text_b = tokenization.convert_to_unicode(str(data[1]))
    label = str(0)
    examples.append(InputExample(guid=guid, text_a=text_a, label=label))
    return examples

data_time_start=time.time()
predict_examples = get_sentence_examples([(rows[0])])
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
# vecotr_in=np.load("data_array_50.npy").astype(np.float32)
# print(vecotr_in[0][4])
# vecotr_in0=np.load("data_array_50.npy")
# print(vecotr_in0[0][4])
# vecotr_in1=np.load("test_data_2.npy")
# with open('./word_file/974_vector_json_lite.json','r',encoding='utf-8') as f:
#     row_data = json.load(f)
#     vecotr_in=np.array(row_data).astype(np.float32)

# vecotr_in=np.tile(vecotr_in1,(974,1)).reshape((974,312)).astype(np.float32)
##############
#############google模型测试输入
# interpreter.set_tensor(input_details[0]['index'], np.array([[101,2339,1555,7213,6121,102]]).astype(numpy.float32))
# interpreter.set_tensor(input_details[1]['index'], np.array([[0,0,0,0,0,0]]).astype(numpy.float32))
# interpreter.set_tensor(input_details[2]['index'],vecotr_in)
###########################
#############brightmart模型测试输入
t1 = time.time()
interpreter.set_tensor(input_details[0]['index'], input_ids)
interpreter.set_tensor(input_details[1]['index'], input_mask)
interpreter.set_tensor(input_details[2]['index'], segment_ids)
# interpreter.set_tensor(input_details[3]['index'], vecotr_in)

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



for i in range(1,len(rows)):
        print('#############'+str(i))
        predict_examples = get_sentence_examples([(rows[i])])
        features = convert_examples_to_features(predict_examples, ['0', '1'], 32,
                                                        tokenization.FullTokenizer(vocab_file='./albert_config/vocab.txt', do_lower_case=True))
        input_ids = [features[0].input_ids]
        input_mask = [features[0].input_mask]
        segment_ids = [features[0].segment_ids]
        # print('input_ids:',input_ids)

        interpreter.set_tensor(input_details[0]['index'], input_ids)
        interpreter.set_tensor(input_details[1]['index'], input_mask)
        interpreter.set_tensor(input_details[2]['index'], segment_ids)
        interpreter.invoke()
        prediction_1 = interpreter.get_tensor(output_details[0]['index'])
        # prediction_1_float16=prediction_1.astype(np.float16)
        # dic_index[i] = prediction_1_float32.tolist()[0]
        predictor_0 = np.r_[predictor_0, prediction_1]

tmp_list=predictor_0.tolist()
saveData=[]
for data in tmp_list:
    alist=[round(i,6) for i in data]
    saveData.append(alist)
with open('./word_file/data_1000.json','w+',encoding='utf8') as file:
    file.write(json.dumps(saveData))