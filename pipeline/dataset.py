# Imports
import nltk
from nltk.corpus import reuters

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


# Main function
def main():

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


if __name__ == '__main__':
    main()
