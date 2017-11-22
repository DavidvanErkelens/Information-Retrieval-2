# Imports
import nltk
from nltk.corpus import reuters
from sklearn.manifold import TSNE
from gensim.models import Word2Vec, LdaModel
from gensim import corpora, matutils
import gensim
import matplotlib.pyplot as plt
from random import randint
import numpy as np


# Available function for reuters:
# reuters.fileids()         =>  files in format 'test/X' or 'training/X'
# reuters.categories()      =>  categories in the dataset
# reuters.categories(file)  =>  categories for a certain file
# reuters.fileids(cat)      =>  files for a certain category
# reuters.words()           =>  all words in the dataset
# reuters.words(file)       =>  words in a file
# 
# Lowercase words are contents of the news article, uppercase words are
# titles of the news articles

# Load the dataset
def loadDataset():

    print("Loading dataset...")

    # Variables that will be filled
    train_docs = []
    test_docs = []
    word2token = {}
    token2word = {}
    num_docs = 0
    num_words = 0
    
    # Data set should be available
    nltk.download('reuters')

    # Fetch the documents
    documents = reuters.fileids()

    # Loop over documents
    for doc in documents:

        # Get the type (test or training) and the file ID
        type, id = doc.split('/')
        
        # This is a test document
        if type == 'test':

            # Skip test documents for now
            pass

        # We're dealing with a training document
        else:
            
            # Increment document counter
            num_docs += 1

            # Get words for this document
            words = reuters.words(doc)

            # Loop over words
            for word in words:

                # Did we see this word before?
                if word not in word2token:

                    # Increment word counter
                    num_words += 1

                    # Store in dictionaries
                    word2token[word] = num_words
                    token2word[num_words] = word

            # Store document
            train_docs.append({'id' : num_docs, 'words' : words, 'tokens' : [word2token[x] for x in words]})

    # Return values
    return train_docs, word2token, token2word, num_docs, num_words


# Simple word2vec test
def word2vec(train_docs):

    print("Training word2vec model...")

    # Get sentences from the training docs
    sentences = [x['words'] for x in train_docs]

    # Initialize model
    model = Word2Vec(min_count = 2, size = 50, window = 4)

    # Create vocabulary
    model.build_vocab(sentences)
    
    # Basic test
    # print(model.wv.most_similar(positive=['man']))
    model.save('word2vec')

    print ("word2vec model saved as 'word2vec'")


# Simple LSI test
def lsi(corpus, dictionary):
    # More info on paramters: https://radimrehurek.com/gensim/models/lsimodel.html
    lsi = LsiModel(
        corpus=corpus,        # corpus used to train the model
        num_topics=200,       # the number of latent dimensions
        id2word=dictionary,   
        chunksize=20000,      # Training proceeds in chunks of chunksize documents. Tradeoff between speed and memory
        decay=0.9,            # < 1.0 causes re-orientation towards new data trends in the input document stream
        distributed=False,    # enable distributed computing
        onepass=True,         # set to false to force multi-pass stochastic algorithm
        power_iters=3,        # higher improves accuracy, but lowers performance
        extra_samples=150     # influence on stochastic multi-pass algorithm
    )

    lsi.save('lsi_model')

# Simple LDA test
def lda(corpus, dictionary):
    lda = LdaModel(
        corpus=corpus, 
        num_topics=200, 
        id2word=dictionary, 
        distributed=False, 
        chunksize=2000, 
        passes=5, 
        update_every=1, 
        alpha='symmetric', 
        eta=None, 
        decay=0.5, 
        offset=1.0, 
        eval_every=100, 
        iterations=50, 
        gamma_threshold=0.001, 
        minimum_probability=0.001, 
        random_state=None, 
        ns_conf={}, 
        minimum_phi_value=0.01, 
        per_word_topics=False
    )

    lda.save('lda_model')

# TSNE test
def tsne_word2vec(model):

    print("Running t-SNE")

    # Fetch the documents
    documents = reuters.fileids()

    # Store data
    num_docs = 0
    num_categories = 0
    doc_vecs = []
    doc_cats = []
    doc_colors = []
    colors = []
    categories = {}
    category2index = {}

    for i in range(150):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    # Loop over documents
    for doc in documents:

        # Get the type (test or training) and the file ID
        type, id = doc.split('/')
        
        # This is a test document
        if type == 'test':

            num_docs += 1

            # Get words for this document
            words = reuters.words(doc)

            word_vecs = []

            # Loop over words
            for word in words:

                # Is this word in our vocab?
                if word in model.wv.vocab:

                    word_vecs.append(model[word])

        
            doc_vecs.append(np.mean(word_vecs, axis=0))
            
            # Get category
            category = reuters.categories(doc)[0]
            

            if category not in category2index:
                num_categories += 1
                categories[num_categories] = {'color': colors[num_categories], 'name' : category}
                category2index[category] = num_categories

            doc_cats.append(category2index[category])
            doc_colors.append(categories[category2index[category]]['color'])

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(doc_vecs)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], color=doc_colors)
    plt.show()

# Main function
def main():
    train_docs, _, _, _, _ = loadDataset()  
    # word2vec(train_docs)    
    # sentences = [x['words'] for x in train_docs]
    

    word2vec = Word2Vec.load('word2vec')

    tsne_word2vec(word2vec)



    # Create dictionary and corpus
    # dictionary = corpora.Dictionary(sentences)
    # corpus = [dictionary.doc2bow(text) for text in sentences]
    # lda(corpus, dictionary)

if __name__ == '__main__':
    main()
