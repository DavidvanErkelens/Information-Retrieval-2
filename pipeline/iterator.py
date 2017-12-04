import pickle
import glob
import os.path

# Basic wrapper around triple file
class Triple:
    def __init__(self, id):
        self.contents = pickle.load(open('../data/wiki/triples/' + str(id) + '.p', 'rb'))

    def getTokens(self, doc_id):
        if (doc_id < 0 or doc_id > 2):
            return []
        return self.contents['articles'][doc_id]['token']

    def getOutlier(self):
        return self.contents['outlier']

# Basic iterator over triples
class TripleIterator:
    def __init__(self):
        
        # Get max file
        files = glob.glob('../data/wiki/triples/*.p')
        max_id = max([int(x.split('/')[-1].split('.')[0]) for x in files]) # Long live Python 
        self.max = max_id

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        while not os.path.isfile('../data/wiki/triples/' + str(self.current) + '.p'):
            self.current += 1
            if self.current >= self.max:
                raise StopIteration

        triple = Triple(self.current)

        self.current += 1

        return triple

def main():
    for x in TripleIterator():
        # print ("Outlier: " + str(x.getOutlier()))
        pass

if __name__ == '__main__':
    main()

