import tensorflow as tf

def tree_model_fasttest(x, class_num, dropout, field_num, feat_num, feat_dim, qmsk, imsk, reuse, is_training):
    with tf.variable_scope('wlr', reuse=reuse):
        embedding_param = tf.get_variable(name="embtable", shape=[feat_num, feat_dim])
  
        feas = tf.nn.embedding_lookup(ids=x,params=embedding_param)
        querys = tf.reduce_sum(feas[:,:10,:] * tf.expand_dims(qmsk,-1), 1) / tf.reduce_sum(qmsk,1,keepdims=True)
        items= tf.reduce_sum(feas[:,10:,:] * tf.expand_dims(imsk, -1), 1) / tf.reduce_sum(imsk,1,keepdims=True)

        feas = tf.concat([querys, items], 1)

        fc = tf.layers.dense(feas, 128)
        fc = tf.nn.relu(fc)
        fc = tf.layers.dropout(fc, rate=dropout, training=is_training)
        fc = tf.layers.dense(fc, 64)
        fc = tf.nn.relu(fc)
        fc = tf.layers.dropout(fc, rate=dropout, training=is_training)
        fc = tf.layers.dense(fc, 32)
        fc = tf.nn.relu(fc)

        logits = tf.layers.dense(fc, class_num)
        res = tf.nn.sigmoid(logits)
    return res
