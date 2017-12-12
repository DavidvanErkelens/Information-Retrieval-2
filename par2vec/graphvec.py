import random
import time

import numpy as np
import scipy.sparse as ss
import tensorflow as tf

from par2vec.common import *

class GraphVec():
    def __init__(self, corpus=None, vocab_size=10, h_layers = [8, 4],
                 act = tf.nn.relu, dropout=0.0, learning_rate = 1e-3):
        """Geo-Vec model as described in the report model section."""

        self.corpus = corpus
        self.vocab_size = vocab_size
        self.h_layers = h_layers
        self.act = act
        self.dropout = dropout
        self.learning_rate = learning_rate

        # use for plotting
        self._loss_vals, self._acc_vals = [], []

        #placeholders
        s = [self.vocab_size, self.vocab_size]
        self.placeholders = {
            'A_o': tf.sparse_placeholder(tf.float32),
            'L_o': tf.sparse_placeholder(tf.float32),
            'A_i': tf.sparse_placeholder(tf.float32),
            'L_i': tf.sparse_placeholder(tf.float32),
            'idx_i': tf.placeholder(tf.int64),
            'idx_o': tf.placeholder(tf.int64),
            'val_i': tf.placeholder(tf.float32),
            'val_o': tf.placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }

        # model
        self.aux_losses = None
        dummy = sp2tf(ss.eye(self.vocab_size))
        self.init_model(x=dummy)

        #optimizer
        self.init_optimizer()

        #sess
        self.trained = 0
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def init_model(self, x, aux_tasks = None):
        """geo-vec model with variable number of gcn layers. Optional aux_taks
        param is now unimplemented to specify which tasks to add. All aux losses
        should be gathered in a self.aux_losses variable to gather later on."""
        for i, h_layer in enumerate(self.h_layers):
            if i == 0:
                h = self.gcn(x, self.vocab_size, self.h_layers[0], self.act, layer=i,sparse=True)
            elif (i+1) < len(self.h_layers):
                h = self.gcn(h, self.h_layers[i-1], h_layer, self.act, layer=i, seperate=True)
            else:
                self.emb_o, self.emb_i = self.gcn(h, self.h_layers[i-1],
                                             h_layer, act=lambda x: x, layer=i, separate=False)

        # here we can left multiply the last layer h
        # and perform auxilliary tasks.
        posneg_samples_o = tf.gather(self.emb_o, tf.transpose(self.placeholders['idx_o']))
        posneg_samples_i = tf.gather(self.emb_i, tf.transpose(self.placeholders['idx_i']))

        self.recon_o = self.decode(posneg_samples_o)
        self.recon_i = self.decode(posneg_samples_i)

    def gcn(self, x, dim_in, dim_out, act, layer, sparse=False, separate=False):
        """basic graph convolution using a split up adjacency matrix.
        The separation param is to create the final embeddings to reconstruct."""
        w1 = tf.get_variable('w1_{}'.format(layer), shape=[dim_in, dim_out],
                             initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable('w2_{}'.format(layer), shape=[dim_in, dim_out],
                             initializer=tf.contrib.layers.xavier_initializer())

        if sparse:
            x1 = tf.sparse_tensor_dense_matmul(x, w1)
            x2 = tf.sparse_tensor_dense_matmul(x, w2)
        else:
            x1 = tf.matmul(x, w1)
            x2 = tf.matmul(x, w2)

        x1 = tf.sparse_tensor_dense_matmul(self.placeholders['L_o'], x1)
        x2 = tf.sparse_tensor_dense_matmul(self.placeholders['L_i'], x2)

        if separate:
            return self.act(x1), self.act(x2)

        return self.act(x1 + x2)

    def decode(self, x, cap = 1000):
        """simple innerproduct decoder with sigmoid activation to scale
        the edged between 0-1000 (assuming more co-occurances are unlikely)."""
#         print(x)
#         print(x.shape)
#         a_t = x
#         idx = tf.where(tf.not_equal(a_t, 0))
#         # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape() if tensor shape is dynamic
#         x = tf.SparseTensor(idx, tf.gather_nd(a_t, idx), a_t.get_shape())

        x = tf.nn.dropout(x, 1-self.dropout)

#         zero = tf.constant(0, dtype=tf.float32)
#         A_rows = tf.sparse_reduce_sum(tf.sparse_add(self.placeholders['A_o'], sp2tf(-ss.eye(self.vocab_size))), 0)

#         where = tf.not_equal(A_rows, zero)
#         indices = tf.where(where)
#         x = tf.gather_nd(x, tf.transpose(indices))

        x = tf.reshape(tf.matmul(x, tf.transpose(x)), [-1])


        return tf.nn.relu(x)

    def init_optimizer(self):
        """initializes optimizer and computes loss + accuracy. The loss function
        is currently a MSE, due to the fact we are dealing with weighted edges.
        This does not seem ideal, and should be thought about."""
        labels_o = self.recon_o
        labels_i = self.recon_i
#         labels_o = tf.reshape(tf.sparse_tensor_to_dense(
#                                 tf.gather(self.placeholders['A_i'], tf.transpose(self.placeholders['idx_i'])),
#                                 validate_indices=False), [-1])
#         labels_i = tf.reshape(tf.sparse_tensor_to_dense(
#                                 tf.gather(self.placeholders['A_i'], tf.transpose(self.placeholders['idx_i'])),
#                                 validate_indices=False), [-1])

        emb_or = tf.gather(self.emb_o, self.placeholders['idx_o'][:, 0])
        emb_oc = tf.gather(self.emb_o, self.placeholders['idx_o'][:, 1])

        emb_ir = tf.gather(self.emb_i, self.placeholders['idx_i'][:, 0])
        emb_ic = tf.gather(self.emb_i, self.placeholders['idx_i'][:, 1])

        self.recon_o = tf.reduce_sum(tf.multiply(emb_or, emb_oc), 1)
        self.recon_i = tf.reduce_sum(tf.multiply(emb_ir, emb_ic), 1)

        loss_o = tf.losses.mean_squared_error(self.recon_o, self.placeholders['val_o'],
                                             weights=self.get_weights(self.placeholders['val_o']))
        loss_i = tf.losses.mean_squared_error(self.recon_i, self.placeholders['val_i'],
                                             weights=self.get_weights(self.placeholders['val_i']))
        self.loss = loss_o + loss_i

        # gather aux losses and add to total loss
        if self.aux_losses:
            self.loss += self.aux_losses

        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.opt_op = optimizer.minimize(self.loss)

        cp_o = tf.equal(tf.cast(self.recon_o, tf.int32), tf.cast(self.placeholders['val_o'], tf.int32))
        cp_i = tf.equal(tf.cast(self.recon_i, tf.int32), tf.cast(self.placeholders['val_i'], tf.int32))
        correct_prediction = tf.concat([cp_o, cp_i], 0)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def get_weights(self, labels):
        """Compute positive and negative example weights"""
        pos_sum = tf.reduce_sum(labels)
        pos_weight = (128**2 - pos_sum) / pos_sum
        thresh = tf.fill(tf.shape(labels), 1.)
        x = tf.fill(tf.shape(labels), 0.001)
        y = tf.fill(tf.shape(labels), pos_weight)
        return tf.where(tf.less(labels, thresh), x, y)

    def get_feed_dict(self, A_o, A_i, L_o, L_i, idx_i, idx_o, val_o, val_i):
        feed_dict = {self.placeholders['A_o']: A_o,
                     self.placeholders['A_i']: A_i,
                     self.placeholders['L_o']: L_o,
                     self.placeholders['L_i']: L_i,
                     self.placeholders['idx_o']: idx_o,
                     self.placeholders['idx_i']: idx_i,
                     self.placeholders['val_o']: val_o,
                     self.placeholders['val_i']: val_i}
        return feed_dict

    def get_sample(self, batch_size=64, ratio=1.0):
        """get random sample from corpus graph cache"""
        dummy = random.choice(Ls).copy()

        pos_idx_o = np.random.choice(range(len(dummy[0].row)), batch_size)
        pos_idx_i = np.random.choice(range(len(dummy[1].row)), batch_size)

        idx_o = np.array(list(zip(dummy[0].row, dummy[0].col)))[pos_idx_o, :]
        idx_i = np.array(list(zip(dummy[1].row, dummy[1].col)))[pos_idx_i, :]
        val_o = dummy[0].data[pos_idx_o]
        val_i = dummy[1].data[pos_idx_i]

        for i, d in enumerate(dummy):
            dummy[i] = sp2tf(d)

        return dummy, idx_o, idx_i, val_o, val_i

    def train(self, num_epochs = 100, print_freq=50):
        """train op that can be invoked multiple times."""
        tf.set_random_seed(42)
        np.random.seed(42)

        for e in range(num_epochs):
            self.trained += 1
            (A_o, A_i, L_o, L_i), idx_o, idx_i, val_o, val_i = self.get_sample()

            feed_dict = self.get_feed_dict(A_o, A_i, L_o, L_i, idx_o, idx_i, val_o, val_i)

            outs = self.sess.run([self.opt_op, self.loss, self.accuracy], feed_dict=feed_dict)
            avg_loss, avg_acc = outs[1], outs[2]
            self._loss_vals.append(avg_loss)
            self._acc_vals.append(avg_acc)

            print('\r epoch: %d/%d \t loss: %.3f \t avg_acc: %.3f'
                      % (e+1, num_epochs, avg_loss, avg_acc), end='')
            if (e + 1) % print_freq == 0:
                print('')
        else:
            print('----> done training: {} epochs'.format(self.trained))

    def plot(self):
        """Plotting loss function"""
        plt.figure(figsize=(12, 6))
        plt.plot(self._loss_vals, color='red')
        plt.plot(self._acc_vals, color='blue')

        plt.legend(handles=[mpatches.Patch(color='red', label='loss'),
                            mpatches.Patch(color='blue', label='acc')],
                   bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    def get_reconstruction(self, doc = None):
        if doc:
            A_o, A_i, L_o, L_i = Doc2Graph(doc, doc_id).doc2graph()
        else:
            (A_o, A_i, L_o, L_i), idx_o, idx_i, val_o, val_i = self.get_sample()
#             A_o, A_i, L_o, L_i = self.get_sample()

        feed_dict = self.get_feed_dict(A_o, A_i, L_o, L_i, idx_o, idx_i, val_o, val_i)
#         feed_dict = self.get_feed_dict(A_o, A_i, L_o, L_i)
        recon_o, recon_i = self.sess.run([self.recon_o, self.recon_i], feed_dict=feed_dict)
        return A_o, A_i, recon_o, recon_i

    def get_embeddings(self, doc = None, doc_id = None):
        if doc:
            A_o, A_i, L_o, L_i = Doc2Graph(doc, doc_id).doc2graph()
        else:
            (A_o, A_i, L_o, L_i), idx_o, idx_i, val_o, val_i = self.get_sample()
#             A_o, A_i, L_o, L_i = self.get_sample()

        feed_dict = self.get_feed_dict(A_o, A_i, L_o, L_i, idx_o, idx_i, val_o, val_i)

#         feed_dict = self.get_feed_dict(A_o, A_i, L_o, L_i, )
        emb_o, emb_i = self.sess.run([self.emb_o, self.emb_i], feed_dict=feed_dict)
        return A_o, A_i, emb_o, emb_i
