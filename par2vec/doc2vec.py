'''
Edit of doc2vec implementation by Zichen Wang (wangzc921@gmail.com)
Source: https://github.com/wangz10/tensorflow-playground/blob/master/doc2vec.py
@references:
http://arxiv.org/abs/1405.4053
'''
from __future__ import absolute_import
from __future__ import print_function

import os
import math
import random
import json
import collections
from itertools import compress

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances

class Doc2Vec(BaseEstimator, TransformerMixin):

    def __init__(self, batch_size=128, window_size=8, 
        concat=True,
        architecture='pvdm', embedding_size_w=128, 
        embedding_size_d=128,
        vocabulary_size=60000,
        document_size=21000,
        loss_type='nce_loss', n_neg_samples=64,
        optimize='Adagrad', 
        learning_rate=1.0, n_steps=100001):
        # bind params to class
        self.data_index = 0
        self.batch_size = batch_size
        self.window_size = window_size
        self.concat = concat
        self.architecture = architecture
        self.embedding_size_w = embedding_size_w
        self.embedding_size_d = embedding_size_d
        self.vocabulary_size = vocabulary_size
        self.document_size = document_size
        self.loss_type = loss_type
        self.n_neg_samples = n_neg_samples 
        self.optimize = optimize
        self.learning_rate = learning_rate
        self.n_steps = n_steps

        # choose a batch_generator function for feed_dict
        self._choose_batch_generator()
        
        # init all variables in a tensorflow graph
        self._init_graph()

        # create a session
        config = tf.ConfigProto(allow_soft_placement = True)
        self.sess = tf.Session(graph=self.graph, config=config)

    def _generate_batch_pvdm(self, doc_ids, word_ids, batch_size, window_size):
        '''
        Batch generator for PV-DM (Distributed Memory Model of Paragraph Vectors).
        batch should be a shape of (batch_size, window_size+1)
        Parameters
        ----------
        doc_ids: list of document indices 
        word_ids: list of word indices
        batch_size: number of words in each mini-batch
        window_size: number of leading words before the target word 
        '''
        assert batch_size % window_size == 0
        batch = np.ndarray(shape=(batch_size, window_size + 1), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = window_size + 1
        buffer = collections.deque(maxlen=span) # used for collecting word_ids[self.data_index] in the sliding window
        buffer_doc = collections.deque(maxlen=span) # collecting id of documents in the sliding window
        # collect the first window of words
        for _ in range(span):
            buffer.append(word_ids[self.data_index])
            buffer_doc.append(doc_ids[self.data_index])
            self.data_index = (self.data_index + 1) % len(word_ids)

        mask = [1] * span
        mask[-1] = 0 
        i = 0
        while i < batch_size:
            if len(set(buffer_doc)) == 1:
                doc_id = buffer_doc[-1]
                # all leading words and the doc_id
                batch[i, :] = list(compress(buffer, mask)) + [doc_id]
                labels[i, 0] = buffer[-1] # the last word at end of the sliding window
                i += 1
            # move the sliding window  
            buffer.append(word_ids[self.data_index])
            buffer_doc.append(doc_ids[self.data_index])
            self.data_index = (self.data_index + 1) % len(word_ids)

        return batch, labels
    
    def _build_doc_dataset(self, docs, vocabulary_size=20000):
        count = [['UNK', -1]]
        # words = reduce(lambda x,y: x+y, docs)
        words = []
        doc_ids = [] # collect document(sentence) indices
        for i, doc in enumerate(docs):
            doc_ids.extend([i] * len(doc))
            words.extend(doc)

        word_ids, count, dictionary, reverse_dictionary = self._build_dataset(words, vocabulary_size=vocabulary_size)

        return doc_ids, word_ids, count, dictionary, reverse_dictionary


    def _build_dataset(self, words, vocabulary_size=20000):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
        dictionary = dict() # {word: index}
        for word, _ in count:
            dictionary[word] = len(dictionary)
            data = list() # collect index
            unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count # list of tuples (word, count)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reverse_dictionary
        
    def _choose_batch_generator(self):
        if self.architecture == 'pvdm':
            self.generate_batch = self._generate_batch_pvdm

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing:
        input data, variables, model, loss function, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/gpu:0'):
            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size+1])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            # Variables.
            # embeddings for words, W in paper
            self.word_embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size_w], -1.0, 1.0))

            # embedding for documents (can be sentences or paragraph), D in paper
            self.doc_embeddings = tf.Variable(
                tf.random_uniform([self.document_size, self.embedding_size_d], -1.0, 1.0))

            if self.concat: # concatenating word vectors and doc vector
                combined_embed_vector_length = self.embedding_size_w * self.window_size + self.embedding_size_d
            else: # concatenating the average of word vectors and the doc vector 
                combined_embed_vector_length = self.embedding_size_w + self.embedding_size_d

            # softmax weights, W and D vectors should be concatenated before applying softmax
            self.weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, combined_embed_vector_length],
                    stddev=1.0 / math.sqrt(combined_embed_vector_length)))
            # softmax biases
            self.biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Model.
            # Look up embeddings for inputs.
            # shape: (batch_size, embeddings_size)
            embed = [] # collect embedding matrices with shape=(batch_size, embedding_size)
            if self.concat:
                for j in range(self.window_size):
                    embed_w = tf.nn.embedding_lookup(self.word_embeddings, self.train_dataset[:, j])
                    embed.append(embed_w)
            else:
                # averaging word vectors
                embed_w = tf.zeros([self.batch_size, self.embedding_size_w])
                for j in range(self.window_size):
                    embed_w += tf.nn.embedding_lookup(self.word_embeddings, self.train_dataset[:, j])
                embed.append(embed_w)
                    
            embed_d = tf.nn.embedding_lookup(self.doc_embeddings, self.train_dataset[:, self.window_size])
            embed.append(embed_d)
            # concat word and doc vectors
            self.embed = tf.concat(embed, 1)

            # Compute the loss, using a sample of the negative labels each time.
            if self.loss_type == 'sampled_softmax_loss':
                loss = tf.nn.sampled_softmax_loss(self.weights, self.biases, self.train_labels,
                    self.embed, self.n_neg_samples, self.vocabulary_size)
            elif self.loss_type == 'nce_loss':
                loss= tf.nn.nce_loss(self.weights, self.biases, self.train_labels, 
                    self.embed, self.n_neg_samples, self.vocabulary_size)
            self.loss = tf.reduce_mean(loss)

            # Optimizer.
            if self.optimize == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
            elif self.optimize == 'SGD':
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            norm_w = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings), 1, keep_dims=True))
            self.normalized_word_embeddings = self.word_embeddings / norm_w

            norm_d = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = self.doc_embeddings / norm_d

            # init op 
            self.init_op = tf.global_variables_initializer()
            # create a saver 
            self.saver = tf.train.Saver()


    def _build_dictionaries(self, docs):
        '''
        Process tokens and build dictionaries mapping between tokens and 
        their indices. Also generate token count and bind these to self.
        '''

        doc_ids, word_ids, count, dictionary, reverse_dictionary = self._build_doc_dataset(docs, 
            self.vocabulary_size)
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary
        self.count = count
        print('Building dictionary')
        print('vocab size: ', len(dictionary))
        print('doc size:   ', len(docs))
        
        return doc_ids, word_ids


    def fit(self, docs):
        '''
        words: a list of words. 
        '''
        # pre-process words to generate indices and dictionaries
        doc_ids, word_ids = self._build_dictionaries(docs)

        # with self.sess as session:
        session = self.sess

        session.run(self.init_op)

        average_loss = 0
        print("Initialized")
        for step in range(self.n_steps):
            batch_data, batch_labels = self.generate_batch(doc_ids, word_ids,
                self.batch_size, self.window_size)
            feed_dict = {self.train_dataset : batch_data, self.train_labels : batch_labels}
            
            op, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0

        # bind embedding matrices to self
        self.word_embeddings = session.run(self.normalized_word_embeddings)
        self.doc_embeddings = session.run(self.normalized_doc_embeddings)

        return self

    def save(self, path):
        '''
        To save trained model and its params.
        '''
        save_path = self.saver.save(self.sess, 
            os.path.join(path, 'model.ckpt'))
        # save parameters of the model
        params = self.get_params()
        json.dump(params, 
            open(os.path.join(path, 'model_params.json'), 'wb'))
        
        # save dictionary, reverse_dictionary
        json.dump(self.dictionary, 
            open(os.path.join(path, 'model_dict.json'), 'wb'), 
            ensure_ascii=False)
        json.dump(self.reverse_dictionary, 
            open(os.path.join(path, 'model_rdict.json'), 'wb'), 
            ensure_ascii=False)

        print("Model saved in file: %s" % save_path)
        return save_path

    def _restore(self, path):
        with self.graph.as_default():
            self.saver.restore(self.sess, path)

    @classmethod
    def restore(cls, path):
        '''
        To restore a saved model.
        '''
        # load params of the modeprint()self.train_datasee
        path_dir = os.path.dirname(path)
        params = json.load(open(os.path.join(path_dir, 'model_params.json'), 'rb'))
        # init an instance of this class
        estimator = Doc2Vec(**params)
        estimator._restore(path)
        # evaluate the Variable embeddings and bind to estimator
        estimator.word_embeddings = estimator.sess.run(estimator.normalized_word_embeddings)
        estimator.doc_embeddings = estimator.sess.run(estimator.normalized_doc_embeddings)
        # bind dictionaries 
        estimator.dictionary = json.load(open(os.path.join(path_dir, 'model_dict.json'), 'rb'))
        reverse_dictionary = json.load(open(os.path.join(path_dir, 'model_rdict.json'), 'rb'))
        # convert indices loaded from json back to int since json does not allow int as keys
        estimator.reverse_dictionary = {int(key):val for key, val in reverse_dictionary.items()}

        return estimator
