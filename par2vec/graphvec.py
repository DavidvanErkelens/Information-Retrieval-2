import numpy as np
import scipy.sparse as ss
import tensorflow as tf
import matplotlib.pyplot as plt

from common import sp2tf, get_lapl


class GraphVec():
    def __init__(self, corpus=None, vocab_size=10, h_layers=[8, 4],
                 act=tf.nn.relu, dropout=0.0, learning_rate=1e-3,
                 pos_sample_size=128, embedding_size_w=128,
                 embedding_size_d=2, n_neg_samples=64,
                 window_size=8, window_batch_size=128):
        """Geo-Vec model as described in the report model section."""

        self.corpus = corpus
        self.vocab_size = vocab_size
        self.h_layers = h_layers
        self.act = act
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.pos_sample_size = pos_sample_size
        self.embedding_size_w = embedding_size_w
        self.embedding_size_d = embedding_size_d
        self.n_neg_samples = n_neg_samples
        self.window_size = window_size
        self.window_batch_size = window_batch_size

        # use for plotting
        self._loss_vals, self._acc_vals = [], []

        # placeholders
        # s = [self.vocab_size, self.vocab_size]
        self.placeholders = {
            'A_o': tf.sparse_placeholder(tf.float32),
            'L_o': tf.sparse_placeholder(tf.float32),
            'A_i': tf.sparse_placeholder(tf.float32),
            'L_i': tf.sparse_placeholder(tf.float32),
            'idx_i': tf.placeholder(tf.int64),
            'idx_o': tf.placeholder(tf.int64),
            'val_i': tf.placeholder(tf.float32),
            'val_o': tf.placeholder(tf.float32),
            'train_dataset': tf.placeholder(tf.int32),
            'train_labels': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }

        # model
        self.aux_losses = None
        dummy = sp2tf(ss.eye(self.vocab_size))
        self.init_model(x=dummy)

        # saver
        self.saver = tf.train.Saver()

        # optimizer
        self.init_optimizer()

        # sess
        self.trained = 0
        # self.sess = tf.Session(graph=self.graph)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def init_model(self, x, aux_tasks=None):
        """geo-vec model with variable number of gcn layers. Optional aux_taks
        param is now unimplemented to specify which tasks to add. All aux losses
        should be gathered in a self.aux_losses variable to gather later on."""
        # self.graph = tf.Graph()
        self.h = []
        # with self.graph.as_default(), tf.device('/gpu:0'):
        for i, h_layer in enumerate(self.h_layers):
            if i == 0:
                self.h.append(self.gcn(x, self.vocab_size, self.h_layers[0], self.act, layer=i, sparse=True))
            elif (i+1) < len(self.h_layers):
                self.h.append(self.gcn(self.h[-1], self.h_layers[i-1], h_layer, self.act, layer=i, seperate=True))
            else:
                self.emb_o, self.emb_i = self.gcn(self.h[-1], self.h_layers[i-1],
                                                  h_layer, act=lambda x: x, layer=i, separate=True)

        self.word_embeddings = tf.Variable(
            tf.random_uniform([self.vocab_size, self.embedding_size_w], -1.0, 1.0))

        # concatenating word vectors and doc vector
        combined_embed_vector_length = self.embedding_size_w * self.window_size + self.embedding_size_d*self.h_layers[-2]

        # softmax weights, W and D vectors should be concatenated before applying softmax
        self.weights = tf.Variable(
            tf.truncated_normal([self.vocab_size, combined_embed_vector_length],
                                stddev=1.0 / np.sqrt(combined_embed_vector_length)))

        # softmax biases
        self.biases = tf.Variable(tf.zeros([self.vocab_size]))

        # Model.
        # Look up embeddings for inputs.
        # shape: (batch_size, embeddings_size)
        embed = []  # collect embedding matrices with shape=(batch_size, embedding_size)
        for j in range(self.window_size):
            embed_w = tf.nn.embedding_lookup(self.word_embeddings, self.placeholders['train_dataset'][:, j])
            embed.append(embed_w)

        self.doc_embeddings = tf.Variable(
            tf.random_uniform([self.embedding_size_d, self.vocab_size], -1.0, 1.0))

        embed_d = tf.expand_dims(tf.reshape(tf.matmul(self.doc_embeddings, self.h[-1]), [-1]), 0)
        embed_d = tf.tile(embed_d, [tf.shape(embed[0])[0], 1])

        embed.append(embed_d)
        # concat word and doc vectors
        self.embed = tf.concat(embed, 1)

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

    def init_optimizer(self):
        """initializes optimizer and computes loss + accuracy. The loss function
        is currently a MSE, due to the fact we are dealing with weighted edges.
        This does not seem ideal, and should be thought about."""
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

        aux_loss = tf.nn.nce_loss(self.weights, self.biases, self.placeholders['train_labels'],
                                  self.embed, self.n_neg_samples, self.vocab_size)
        self.aux_losses = tf.reduce_mean(aux_loss)

        # gather aux losses and add to total loss
        if self.aux_losses is not None:
            self.loss += tf.scalar_mul(0.05, self.aux_losses)

        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.opt_op = optimizer.minimize(self.loss)

        cp_o = tf.equal(tf.cast(self.recon_o, tf.int32),
                        tf.cast(self.placeholders['val_o'], tf.int32))
        cp_i = tf.equal(tf.cast(self.recon_i, tf.int32),
                        tf.cast(self.placeholders['val_i'], tf.int32))
        correct_prediction = tf.concat([cp_o, cp_i], 0)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def get_weights(self, labels):
        """Compute positive and negative example weights"""
        pos_sum = tf.reduce_sum(labels)
        pos_weight = (64**2 - pos_sum) / pos_sum
        thresh = tf.fill(tf.shape(labels), 1.)
        x = tf.fill(tf.shape(labels), 0.001)
        y = tf.fill(tf.shape(labels), pos_weight)
        return tf.where(tf.less(labels, thresh), x, y)

    def get_feed_dict(self, A_o, A_i, L_o, L_i, idx_i, idx_o, val_o, val_i, train_dataset, train_labels):
        feed_dict = {self.placeholders['A_o']: A_o,
                     self.placeholders['A_i']: A_i,
                     self.placeholders['L_o']: L_o,
                     self.placeholders['L_i']: L_i,
                     self.placeholders['idx_o']: idx_o,
                     self.placeholders['idx_i']: idx_i,
                     self.placeholders['val_o']: val_o,
                     self.placeholders['val_i']: val_i,
                     self.placeholders['train_dataset']: train_dataset,
                     self.placeholders['train_labels']: train_labels}
        return feed_dict

    def get_sample(self, ratio=1.0):
        """get random sample from corpus graph cache"""
        doc = [np.array([])]
        while len(doc[0]) < self.window_size:
            r_doc_id = np.random.randint(len(self.corpus['tokenized']))
            doc = [np.array(self.corpus['tokenized'][r_doc_id]).copy()]
        docidx, A_o, A_i, L_o, L_i = get_lapl(doc, self.corpus['word2id']).__next__()

        pos_idx_o = np.random.choice(range(len(A_o.row)), self.pos_sample_size)
        pos_idx_i = np.random.choice(range(len(A_i.row)), self.pos_sample_size)

        idx_o = np.array(list(zip(A_o.row, A_i.col)))[pos_idx_o, :]
        idx_i = np.array(list(zip(A_i.row, A_i.col)))[pos_idx_i, :]
        val_o = A_o.data[pos_idx_o]
        val_i = A_i.data[pos_idx_i]

        dummy = []
        for i, d in enumerate([A_o, A_i, L_o, L_i]):
            dummy.append(sp2tf(d))

        windows = np.copy(np.lib.stride_tricks.as_strided(docidx, (len(docidx)-self.window_size, self.window_size+1),
                                                          2 * docidx.strides))
        np.random.shuffle(windows)
        # print(windows)

        train_dataset = windows[:128, :-1]
        train_labels = windows[:128, -1:]

        return dummy, idx_o, idx_i, val_o, val_i, train_dataset, train_labels

    def train(self, num_epochs=100, print_freq=50, backup_freq=None):
        """train op that can be invoked multiple times."""
        tf.set_random_seed(43)
        np.random.seed(43)

        for e in range(num_epochs):
            self.trained += 1
            (A_o, A_i, L_o, L_i), idx_o, idx_i, val_o, val_i, train_dataset, train_labels = self.get_sample()
            # print(train_labels)

            feed_dict = self.get_feed_dict(A_o, A_i, L_o, L_i, idx_o, idx_i, val_o, val_i, train_dataset, train_labels)

            outs = self.sess.run([self.opt_op, self.loss, self.aux_losses, self.accuracy], feed_dict=feed_dict)
            avg_loss, aux_loss, avg_acc = outs[1], outs[2], outs[3]
            self._loss_vals.append(avg_loss)
            self._acc_vals.append(avg_acc)

            print('\r epoch: %d/%d \t graph loss: %.3f \t aux loss: %.3f \t avg_acc: %.3f'
                  % (e+1, num_epochs, avg_loss, aux_loss, avg_acc), end='')
            if (e + 1) % print_freq == 0:
                print('')

            if backup_freq:
                if (e + 1) % backup_freq == 0:
                    self.save('models/model_{}.ckpt'.format(e + 1))
                    
        else:
            print('----> done training: {} epochs'.format(self.trained))
            self.save('models/model_final.ckpt')

    def plot(self):
        """Plotting loss function"""
        plt.figure(figsize=(12, 6))
        plt.plot(self._loss_vals, color='red')
        plt.plot(self._acc_vals, color='blue')

        plt.legend(handles=[mpatches.Patch(color='red', label='loss'),
                            mpatches.Patch(color='blue', label='acc')],
                   bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    def get_reconstruction(self, doc=None):
        if doc:
            # A_o, A_i, L_o, L_i = Doc2Graph(doc, doc_id).doc2graph()
            (A_o, A_i, L_o, L_i), idx_o, idx_i, val_o, val_i = self.get_sample()
        else:
            (A_o, A_i, L_o, L_i), idx_o, idx_i, val_o, val_i = self.get_sample()

        feed_dict = self.get_feed_dict(A_o, A_i, L_o, L_i, idx_o, idx_i, val_o, val_i)
        recon_o, recon_i = self.sess.run([self.recon_o, self.recon_i], feed_dict=feed_dict)
        return A_o, A_i, recon_o, recon_i

    def get_embeddings(self, doc=None, doc_id=None):
        if doc:
            # A_o, A_i, L_o, L_i = 1#Doc2Graph(doc, doc_id).doc2graph()
            (A_o, A_i, L_o, L_i), idx_o, idx_i, val_o, val_i = self.get_sample()
        else:
            (A_o, A_i, L_o, L_i), idx_o, idx_i, val_o, val_i = self.get_sample()

        feed_dict = self.get_feed_dict(A_o, A_i, L_o, L_i, idx_o, idx_i, val_o, val_i)

        emb_o, emb_i = self.sess.run([self.emb_o, self.emb_i], feed_dict=feed_dict)
        return A_o, A_i, emb_o, emb_i

    def save(self, file_name):
        print('Saving model: ', file_name)
        self.saver.save(self.sess, file_name)

    def load(self, file_name):
        print('Loading model: ', file_name)
        self.saver.restore(sess, file_name)
