import tensorflow as tf
def getinout(input_checkpoint):
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    with tf.Session() as sess:
        file = open('./nodes_albert_lcqmc.txt', 'a+')

        for n in tf.get_default_graph().as_graph_def().node:
            file.write(n.name + '\n')

        file.close()
getinout('./albert_lcqmc_finall/model.ckpt-3730')