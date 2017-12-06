import pickle
import glob
import os.path

# Basic wrapper around triple file
class Triple:
    def __init__(self, id):
        self.id = id
        self.contents = pickle.load(open('../data/wiki/triplets/' + str(id) + '.p', 'rb'))

    def getTokens(self, doc_id):
        if (doc_id < 0 or doc_id > 2):
            return []
        return self.contents['articles'][doc_id]['tokens']

    def getIDs(self, doc_id):
        if (doc_id < 0 or doc_id > 2):
            return []
        return self.contents['articles'][doc_id]['ids']

    def getOutlier(self):
        return self.contents['outlier']

    # Function used to fix tokenize error
    def update(self, id2word):
        for x in range(0,3):
            self.contents['articles'][x]['ids'] = self.contents['articles'][x]['tokens']
            self.contents['articles'][x]['tokens'] = [id2word[y] for y in self.contents['articles'][x]['ids']  if y > 0]

        pickle.dump(self.contents, open('../data/wiki/triplets/fixed/' + str(self.id) + '.p', 'wb'))


# Basic iterator over triplets
class TripleIterator:
    def __init__(self):
        # Get max file
        files = glob.glob('../data/wiki/triplets/*.p')
        max_id = max([int(x.split('/')[-1].split('.')[0]) for x in files]) # Long live Python 
        self.max = max_id

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        while not os.path.isfile('../data/wiki/triplets/' + str(self.current) + '.p'):
            self.current += 1
            if self.current >= self.max:
                raise StopIteration

        triple = Triple(self.current)

        self.current += 1

        return triple

def main():
    # id2word = pickle.load(open('../data/wiki/id2word.p', 'rb'))

    for x in TripleIterator():
        # x.update(id2word)
        pass

if __name__ == '__main__':
    main()

