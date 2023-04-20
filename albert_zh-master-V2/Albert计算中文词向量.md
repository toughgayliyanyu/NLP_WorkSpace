# Albert-Chinese-Sentence-Encoder（基于albert_tiny_zh模型提取中文词向量）

基于albert_tiny_zh模型提取中文词向量

## 功能

通过albert_tiny_zh中间层计算出来的词向量，进行相似比较。实现在词库中搜索与输入最为相似的多个候选词。

## 主要代码说明

![代码目录](D:\PythonWorkSpace\MyPythonProject\albert_zh-master-V2\Albert-Chinese-Sentence-Encoder.assets\代码目录-1681981802702-2.jpg)

&emsp;&emsp;albert_config：词表和albert_tiny_zh模型json配置文件

&emsp;&emsp;albert_tiny_zh：预训练模型

&emsp;&emsp;mrpc_output：将预训练模型权重由float32转float16后的模型

&emsp;&emsp;output_litemodel：pb模型换lite模型后保存的路径

&emsp;&emsp;output_pbmodel：ckpt转pb模型后保存的路径

&emsp;&emsp;resaved_albert_tiny：修改与训练模型获取词向量，并重新保存模型。

&emsp;&emsp;resaved_albert_tiny_vectorall_sorted_1000：词向量和1000维词库计算相似度模型

&emsp;&emsp;1-test_model_load.py：获取预训练模型模型计算词向量的代码

&emsp;&emsp;2-test_model_load_3-sort-1000vectors.py：计算词向量和词库相似度的代码

&emsp;&emsp;3-ckpt_2_pb.py：ckpt转pb的代码

&emsp;&emsp;4-test_pb_moel.py：测试转换后的pb模型

&emsp;&emsp;5-pb_2_lite.py：pb模型转lite代码（输出进行int8量化）

&emsp;&emsp;5-pb_2_lite-normal.py：pb转lite代码（权重进行int8量化）

&emsp;&emsp;6-test_lite_ciku2vec.py：通过lite模型将中文词库转词向量

&emsp;&emsp;6-test_lite_model.py：测试lite模型

## 环境要求

主要使用tensorflow1.15.0

5-pb_2_lite.py：pb模型转lite代码（输出进行int8量化）需要使用tensorflow2.5.0

##详细说明

输入节点信息：

input_ids：dtype=tf.int32, shape=[1, 32]    例如：[[101, 1825, 7032, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

input_mask：dtype=tf.int32, shape=[1, 32]   例如：[[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

segment_ids：dtype=tf.int32, shape=[1, 32]  例如：[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

vectors_all：dtype=tf.float32, shape=[1000, 312]  （是1000个词对应的词向量）

输出节点信息：

bert/pooler/dense/Tanh：dtype=tf.float32, shape=[1, 312]（词向量）

concat：dtype=tf.float32, shape=[2, 1000]（一个词与1000个词计算的相似度以及相似度索引的排序）

## 主要代码说明

1：1-test_model_load.py

​		加载./albert_tiny_zh（ckpt）模型进行词向量计算和重新保存模型。输入（input_ids、input_mask、segment_ids）输出（bert/pooler/dense/Tanh：shape[1,312]）。

2：2-test_model_load_3-sort-1000vectors.py

​		增加多行词向量输入（input_ids、input_mask、segment_ids，**vectors_all**），加载./albert_tiny_zh模型进行预测和重新保存模型。输出（concat：shape[2,1000]）

3：ckpt模型转pb

使用如下命令：

freeze_graph --input_checkpoint=./resaved_albert_tiny/resaved_albert_tiny --output_graph=./output_pbmodel/resaved_albert_tiny.pb --output_node_names=bert/pooler/dense/Tanh --checkpoint_version=1 --input_meta_graph=./resaved_albert_tiny/resaved_albert_tiny.meta --input_binary=true

参数说明：

​		input_checkpoint：保存的ckpt模型路径

​		output_graph：pb模型保存位置

​		output_node_names：输出节点名（对于三个输入节点对应的输出节点：bert/pooler/dense/Tanh。对于四个节点输出节点：concat）

​		checkpoint_version=1

​		input_meta_graph：保存的ckpt模型meta路径

​		input_binary=true

或者使用：3-ckpt_2_pb.py

4：4-test_pb_moel.py

​		测试转换后的pb模型

5：5-pb_2_lite-normal.py

​		将pb模型转lite模型

6：6-test_lite_model.py

​		测试lite模型

​		