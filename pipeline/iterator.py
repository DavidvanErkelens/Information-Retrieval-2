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

    # Keep track of totals
    passed = 0
    correct = 0

    for x in TripleIterator():
        # print ("Outlier: " + str(x.getOutlier()))

        actual_outlier = x.getOutlier()

        doc0 = x.getTokens(0)
        doc1 = x.getTokens(1)
        doc2 = x.getTokens(2)

        # Pass paragraphs though model and calculate distance metric between them
        # Dummy data..
        dist01 = 0
        dist12 = 0
        dist02 = 0

        # Get the detected outlier
        detected_outlier = 0

        if dist01 < dist02 && dist01 < dist12:
            detected_outlier = 2

        if dist02 < dist01 && dist02 < dist12:
            detected_outlier = 1

        # Correct?
        if actual_outlier == detected_outlier:
            correct += 1

        passed += 1

    print("Correct: " + str(correct / float(passed)))

if __name__ == '__main__':
    main()

