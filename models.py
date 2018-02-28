import tensorflow as tf
from settings import *
import numpy as np


class Model(object):
    def __init__(self, is_training=True):
        # input
        in_x = tf.placeholder(dtype=tf.int64, shape=[BATCH_SIZE, SEQUENCE_LEN], name='in_x')  # sentences
        in_e1 = tf.placeholder(dtype=tf.int64, shape=[BATCH_SIZE], name='in_e1')
        in_e2 = tf.placeholder(dtype=tf.int64, shape=[BATCH_SIZE], name='in_e2')
        in_dist1 = tf.placeholder(dtype=tf.int64, shape=[BATCH_SIZE, SEQUENCE_LEN], name='in_dist1')
        in_dist2 = tf.placeholder(dtype=tf.int64, shape=[BATCH_SIZE, SEQUENCE_LEN], name='in_dist2')
        in_y = tf.placeholder(dtype=tf.int64, shape=[BATCH_SIZE], name='in_y')  # relations

        self.inputs = (in_x, in_e1, in_e2, in_dist1, in_dist2, in_y)

        # embeddings
        word_embeddings = np.load(WORD_EMBEDDINGS_PATH)
        initializer = tf.truncated_normal_initializer(stddev=0.1)
        embed = tf.get_variable(initializer=word_embeddings, dtype=tf.float64, name='word_embed')
        pos1_embed = tf.get_variable(initializer=initializer, shape=[NB_DISTANCES, POSITION_EMBED_SIZE], name='position1_embed')
        pos2_embed = tf.get_variable(initializer=initializer, shape=[NB_DISTANCES, POSITION_EMBED_SIZE], name='position2_embed')
        rel_embed = tf.get_variable(initializer=initializer, shape=[NB_RELATIONS, NB_FILTERS], name='relation_embed')

        # embedding lookup
        e1 = tf.nn.embedding_lookup(embed, in_e1, name='e1')  # BATCH_SIZE, WORD_EMBED_SIZE
        e2 = tf.nn.embedding_lookup(embed, in_e2, name='e2')  # BATCH_SIZE, WORD_EMBED_SIZE
        x = tf.nn.embedding_lookup(embed, in_x, name='x')  # BATCH_SIZE, SEQUENCE_LEN, WORD_EMBED_SIZE
        dist1 = tf.nn.embedding_lookup(pos1_embed, in_dist1, name='dist1')  # BATCH_SIZE, SEQUENCE_LEN, POSITION_SIZE
        dist2 = tf.nn.embedding_lookup(pos2_embed, in_dist2, name='dist2')  # BATCH_SIZE, SEQUENCE_LEN, POSITION_SIZE
        x_concat = tf.concat([x, dist1, dist2], -1)  # BATCH_SIZE, SEQUENCE_LEN, WORD_REPRE_SIZE
        x_concat = tf.reshape(x_concat, [BATCH_SIZE, SEQUENCE_LEN, WORD_REPRE_SIZE, 1])

        pooled_outputs = []
        if is_training:
            x_concat = tf.nn.dropout(x_concat, DROPOUT)

        for i, k in enumerate(WINDOW_SIZES):
            with tf.variable_scope("conv-%d" % k):  # , reuse=False
                w = tf.get_variable(initializer=initializer, shape=[k, WORD_REPRE_SIZE, 1, NB_FILTERS], name='weight')
                b = tf.get_variable(initializer=initializer, shape=[NB_FILTERS], name='bias')
                conv = tf.nn.conv2d(x_concat, w, strides=[1, 1, WORD_REPRE_SIZE, 1], padding="SAME")

                h = tf.nn.tanh(tf.nn.bias_add(conv, b), name="h")  # BATCH_SIZE, n, 1, dc

                R = tf.reshape(h, [BATCH_SIZE, SEQUENCE_LEN, NB_FILTERS])
                U = tf.get_variable(initializer=initializer, shape=[NB_FILTERS, NB_FILTERS], name='U')
                G = tf.matmul(tf.reshape(R, [BATCH_SIZE * SEQUENCE_LEN, NB_FILTERS]), U)
                G = tf.matmul(G, tf.transpose(rel_embed))
                G = tf.reshape(G, [BATCH_SIZE, SEQUENCE_LEN, NB_RELATIONS])
                AP = tf.nn.softmax(G, dim=1)  # attention pooling tensor

                wo = tf.matmul(tf.transpose(R, perm=[0, 2, 1]), AP)  # (BATCH_SIZE, NB_FILTERS, NB_RELATIONS)
                wo = tf.reduce_max(wo, axis=-1)  # (BATCH_SIZE, NB_FILTERS)
                pooled_outputs.append(wo)

        h_pool_flat = tf.concat(pooled_outputs, -1)  # (BATCH_SIZE, NB_FILTERS * len(WINDOW_SIZES))

        # entity embedding
        e_flat = tf.concat([e1, e2], -1)
        h_pool_flat = tf.concat([h_pool_flat, e_flat], -1)  # (BATCH_SIZE, NB_FILTERS * len(WINDOW_SIZES) + 2 * WORD_EMBED_SIZE)

        if is_training:
            h_pool_flat = tf.nn.dropout(h_pool_flat, DROPOUT)

        # output
        W_o = tf.get_variable(initializer=initializer, shape=[NB_FILTERS * len(WINDOW_SIZES) + WORD_EMBED_SIZE * 2, NB_RELATIONS], name='w_o')
        b_o = tf.get_variable(initializer=initializer, shape=[NB_RELATIONS], name='b_o')
        scores = tf.nn.xw_plus_b(h_pool_flat, W_o, b_o, name="scores")
        predict = tf.argmax(scores, 1, name="predictions")
        predict = tf.cast(predict, dtype=tf.int64)
        acc = tf.reduce_sum(tf.cast(tf.equal(predict, in_y), dtype=tf.int64))
        self.predict = predict
        self.acc = acc

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=scores,
                                                    labels=tf.one_hot(in_y, NB_RELATIONS))
        )
        l2_loss = tf.nn.l2_loss(W_o)
        l2_loss += tf.nn.l2_loss(b_o)
        l2_loss = L2_REG_LAMBDA * l2_loss

        self.loss = loss + l2_loss

        if not is_training:
            return

        # optimizer
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        global_step = tf.Variable(0, trainable=False, name='global_step')

        self.train_op = optimizer.minimize(self.loss)
        self.reg_op = tf.no_op()
        self.global_step = global_step
