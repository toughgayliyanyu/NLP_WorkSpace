from tensorflow import keras
import tensorflow as tf
from utils import process_w2v_data
corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]

class CBOW(keras.Model):
    def __init__(self,v_dim,emb_dim):
        super().__init__()
        self.v_dim=v_dim
        #Embedding层本质也是一个映射，不过不是映射为on-hot编码，而是映射为一个指定维度的向量，该向量是一个变量，通过学习寻找到最优值
        self.embeddings=keras.layers.Embedding(
            input_dim=v_dim,output_dim=emb_dim,
            embeddings_initializer=keras.initializers.RandomNormal(0.,0.1)
        )
        self.nec_w=self.add_weight(
            name='nec_w',shape=[v_dim,emb_dim],
            initializer=keras.initializers.TruncatedNormal(0.,0.1)
        )
        self.nec_b=self.add_weight(
            name='nec_b',shape=(v_dim,),
            initializer=keras.initializers.Constant(0.1)
        )
        #定义优化器
        self.opt=keras.optimizers.Adam(0.01)
    def call(self, x, training=None, mask=None):
        #计算隐藏曾输出
        o=self.embeddings(x)
        #计算隐藏层沿axis=1方向上的平均值
        o=tf.reduce_mean(o,axis=1)
        return o
    def loss(self,x,y,training=None):
        embedded=self.call(x,training)
        '''
        nec_loss:一种加速计算损失的方法，NCE 的核心思想就是通过学习数据分布样本和噪声分布样本之间的区别，从而发现数据中的一些特性，
        因为这个方法需要依靠与噪声数据进行对比，所以称为“噪声对比估计（Noise Contrastive Estimation）”。更具体来说，NCE 将问题
        转换成了一个二分类问题，分类器能够对数据样本和噪声样本进行二分类，通过最大化同一个目标函数来估计模型参数 。以语言模型为例，利
        用NCE可将从词表中预测某个词的多分类问题，转为从噪音词中区分出目标词的二分类问题。具体如图所示：
        '''
        nec_loss=tf.nn.nce_loss(
                weights=self.nec_w,biases=self.nec_b,labels=tf.expand_dims(y,axis=1),
                inputs=embedded,num_sampled=5,num_classes=self.v_dim
            )
        reducemean=tf.reduce_mean(nec_loss)
        return reducemean
    def step(self,x,y):
        with tf.GradientTape() as tape:
            #输入x和y样本计算损失
            loss=self.loss(x,y,True)
            # 计算损失函数的关于embeddings，nec_w,nec_bde 梯度（导数）
            grads=tape.gradient(loss,self.trainable_variables)
        # 根据梯度更新参数
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy()
def train(model,data):
    for t in range(2500):
        bx,by=data.sample(8)
        loss=model.step(bx,by)
        if t%200==0:
            print('step:{} | loss {}'.format(t,loss))

if __name__=="__main__":
    d=process_w2v_data(corpus,skip_window=2,method="cbow")
    m=CBOW(d.num_word,2)
    train(m,d)
    # print(m.embeddings)