# TF-IDF（Term Frequency - Inverse Document Frequency）
一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章
## TF：词频，反映的是局部信息，在一篇文章或一句话中往往高频词最具代表性。
## IDF：逆文本频率指数，反应了全局信息，往往一些助词或语气词在文中出现的频率也很高，但是这种词并不具有区分力，不能反映文章特征。因此需要引入一种更具区分能力的全局信息（IDF）
## TF*IDF：通过TF和IDF相乘构造出具有代表性的特征信息

# TF*IDF的作用：通过统计学的方法构造文本特征信息
# TF*IDF的数学表达式：
  TF：TF=文档d中词w总数
  IDF：IDF=log(所有文档数/所有文档中词w数)
# TF-IDF改进：
  用 Log，也就是对数函数，对 TF 进行变换，就是一个不让 TF 线性增长的技巧。具体来说，人们常常用 1+Log(TF) 这个值来代替原来的 TF 取值。在这样新的计算下，假设 “Car” 出现一次，新的值是 1，出现 100 次，新的值是 5.6，而出现 200 次，新的值是 6.3。很明显，这样的计算保持了一个平衡，既有区分度，但也不至于完全线性增长。
# TF-IDF应用：
（1）搜索引擎；（2）关键词提取；（3）文本相似性；（4）文本摘要
# 参考：
https://blog.csdn.net/weixin_43734080/article/details/122226507?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-122226507-blog-81486700.pc_relevant_multi_platform_whitelistv3&spm=1001.2101.3001.4242.1&utm_relevant_index=3
