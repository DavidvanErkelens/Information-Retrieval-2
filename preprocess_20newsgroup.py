import numpy as np
from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup
import re
import urllib
import pickle

if False:
    # Load docs
    docs = []
    topic_str = []
    topiclist = [x for x in listdir('data/20_newsgroup/') if not x.startswith('20newsgroup')]
    for topic in topiclist:
        for datafile in sorted(listdir('data/20_newsgroup/{}/'.format(topic))):
            with open('data/20_newsgroup/{}/{}'.format(topic, datafile), 'r', encoding='latin-1') as f:
                docs.append(f.read().replace('\n', ' '))
                topic_str.append([topic])

    # Preprocess topics
    topic2id = dict(zip(topiclist, np.arange(len(topiclist))))
    id2topic = {v: k for k, v in topic2id.items()}
    topics = [[topic2id[x] for x in y] for y in topic_str]

    # Preprocess documents
    splitted_docs = [re.sub('[^a-zA-Z]+', ' ', doc) for doc in docs]
    splitted_docs = [doc.split(' ') for doc in splitted_docs]
    splitted_docs = [[word.lower() for word in doc if word != ''] for doc in splitted_docs]

    words = [x for y in splitted_docs for x in y]
    unique_words, unique_words_c = np.unique(words, return_counts=True)
    _, unique_words_sort = zip(*sorted(zip(unique_words_c, unique_words), reverse=True))

    # Tokenize
    TOP_WORDS = 50000
    word2id = dict(zip(unique_words[:TOP_WORDS], np.arange(1, len(unique_words[:TOP_WORDS]) + 1)))
    word2id['<unk>'] = 0
    id2word = {v: k for k, v in word2id.items()}
    tokenized = [[word2id[word] if word in word2id else 0 for word in doc] for doc in splitted_docs]

    # Save tokenized reuters
    np.save('data/20_newsgroup/20newsgroup_topics.npy', topics)
    np.save('data/20_newsgroup/20newsgroup_topic2id.npy', topic2id)
    np.save('data/20_newsgroup/20newsgroup_id2topic.npy', id2topic)
    np.save('data/20_newsgroup/20newsgroup_word2id.npy', word2id)
    np.save('data/20_newsgroup/20newsgroup_id2word.npy', id2word)
    np.save('data/20_newsgroup/20newsgroup_tokenized.npy', tokenized)

# Load tokenized reuters
topic2id = np.load('data/20_newsgroup/20newsgroup_topic2id.npy').item(0)
id2topic = np.load('data/20_newsgroup/20newsgroup_id2topic.npy').item(0)
topics = list(np.load('data/20_newsgroup/20newsgroup_topics.npy'))

word2id = np.load('data/20_newsgroup/20newsgroup_word2id.npy').item(0)
id2word = np.load('data/20_newsgroup/20newsgroup_id2word.npy').item(0)
tokenized = list(np.load('data/20_newsgroup/20newsgroup_tokenized.npy'))

print(max([len(x) for x in tokenized]))
print(len(word2id))

# Example document
tokenized_doc = tokenized[0]
regular_doc = ' '.join([id2word[x] for x in tokenized_doc])
tokenized_topic = topics[0]
regular_topic = ' '.join([id2topic[x] for x in tokenized_topic])

# Create triplets
triplets = []
for i in range(20):
    same = np.where(np.array(topics)[:,0]==i)[0]
    other = np.where(np.array(topics)[:,0]!=i)[0]
    np.random.shuffle(same)
    np.random.shuffle(other)
    for j in range(1000):
        triplets.append([i, same[j%len(same)], other[j%len(other)]])

pickle.dump(triplets, open('data/20_newsgroup/20newsgroup_triplets.p', 'wb'))

# Create big raw file
bigdocs = '\n'.join([' '.join([id2word[word_id] for word_id in doc]) for doc in tokenized])
with open('data/20_newsgroup/raw.txt', 'w') as f:
    f.write(bigdocs)
