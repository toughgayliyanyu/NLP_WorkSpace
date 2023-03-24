# CBOW(词袋算法)

通过左右词预测中间词的算法。例如：

我爱中华人民共和国

input：我爱，华人  output：中

input：爱中，人民  output：华

input：爱中，人民  output：华



## 算法流程

![cbow1](C:\Users\18800\Desktop\cbow1.png)

![在这里插入图片描述](D:\PythonWorkSpace\GitWorkSpace\NLP\NLP_WorkSpace\CBOW\README.assets\watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBARmVhdGhlcl83NA==,size_20,color_FFFFFF,t_70,g_se,x_16.png)

*第一步：计算输入层

通过one-hot或者词表字典的方式构造输入，窗口大小设为2，意思是选择前后两个词。所以输入的维度是4例如：

| 单词     | one-hot编码 |
| -------- | ----------- |
| i        | [1,0,0,0,0] |
| drink    | [0,1,0,0,0] |
| coffee   | [0,0,1,0,0] |
| everyday | [0,0,0,1,0] |
| morning  | [0,0,0,0,1] |
本文中使用的是另外一种方式通过vocab词表构造输入，例如：

vocab=[‘我’，‘爱’，‘中’，‘华’，‘人’，‘民’，‘共’，‘和’，‘国’]

indexs=[0,1,2,3,4,5,6,7,8]

input：[我,爱,华,人]   编码后：[0,1,3,4]

input：[爱,中,人,民]   编码后：[1,2,4,5]

input：[中,华,民,共]   编码后：[2,3,5,6]

当词非常多的时候，与索引的方式对比，one-hot方式构造出来的输入维度会很大。

*第二步：计算隐藏层

隐藏层权重矩阵是一个V 行N列的，v代表语料库中词的个数，N 是一个任意数字，即最后得到的词向量维度为 N。每个 input vector 分别乘以 W 可以分别得到维度为 N 的词向量，然后再求平均值得到隐藏层向量。

隐藏层向量乘 W’ （ N 行 V 列），得到一个维度为 V 的向量。

*第三步：计算隐藏层

输出层是一个 softmax 回归分类器，它的每个结点将会输出一个0-1之间的值（概率），这些所有输出层神经元结点的概率之和为1。

![在这里插入图片描述](D:\PythonWorkSpace\GitWorkSpace\NLP\NLP_WorkSpace\CBOW\README.assets\76e0bcb207624ac9839774ba2de34bb3.png)

## 参考

1. [(252条消息) Word2Vec之CBOW详解_Feather_74的博客-CSDN博客](https://blog.csdn.net/qq_44997147/article/details/120875909)
2. https://mofanpy.com/tutorials/machine-learning/nlp/cbow
3. https://github.com/MorvanZhou/NLP-Tutorials/blob/master/CBOW.py