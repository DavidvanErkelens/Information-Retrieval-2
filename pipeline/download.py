# Imports
import urllib.request
import pypandoc
import lxml.etree as ET
from bs4 import BeautifulSoup
import re
import pickle

# Main function
def main():

    # Load word IDs
    word2id = pickle.load(open('../data/wiki/id2token.p', 'rb'))
    id2word = pickle.load(open('../data/wiki/id2word.p', 'rb'))

    # Open file with Wikipedia triplets, downloaded from http://cs.stanford.edu/ ̃quocle/triplets-data.tar.gz
    with open('wikipedia_2014_09_27_examples.txt') as file:
        
        # Read contents of the line
        contents = file.readlines()

        # Number of processed triplets
        processed = 0
        iterated = 0

        # Loop over lines
        for line in contents:

            # Update iterated
            iterated += 1

            # Skip commented lines
            if line[0] == '#':
                continue

            # Number of processed articles
            num_articles = 0

            # Save categories per article
            linecats = {}

            # Is this line valid (i.e. do all articles exist and are they parsable)?
            validline = True

            # Store triple
            triple = {}

            # Store articles
            triple['articles'] = {}

            # Prevent errors
            try:


                # Loop over articles
                for item in line.split():

                    # Store article
                    article = {}

                    # Get XML export URL
                    url = item.replace('http://en.wikipedia.org/wiki/', 'https://en.wikipedia.org/wiki/Special:Export/')
                    opener = urllib.request.build_opener()
                    
                    # Parse XML export and get root node
                    tree = ET.parse(opener.open(url))
                    root = tree.getroot()

                    # Try to get the page contents
                    page = root.find('{http://www.mediawiki.org/xml/export-0.10/}page')

                    # If that does not exist, the page is removed and the line is therefore not valid
                    if page is None:
                        validline = False
                        break

                    # Get the content of the page
                    revision = page.find('{http://www.mediawiki.org/xml/export-0.10/}revision')
                    text = revision.find('{http://www.mediawiki.org/xml/export-0.10/}text')
                    content = ET.tostring(text, encoding='ascii').decode('ascii')

                    # Try to convert it to plain text using a Python wrapper around Pandoc
                    try:
                        converted = pypandoc.convert_text(content, 'plain', format='mediawiki').encode('ascii', 'ignore').decode('ascii')
                    except RuntimeError:

                        # We can not convert the text, so we mark this line as invalid
                        validline = False
                        break

                    # Get all categories from the mediawiki format
                    cats = re.findall('\[\[Category:(.+?)\]\]', content)

                    # Store a set of the articles
                    linecats[num_articles] = set(cats)

                    # Cleanup the last bits of HTML from the plain text
                    clean = BeautifulSoup(converted, "lxml").text

                    # Store text
                    article['text'] = clean

                    # Store URL
                    article['url'] = item

                    # Store tokenized article
                    tokenized = [word2id[x] if x in word2id else 0 for x in clean.split()]
                    article['tokens'] = tokenized
                    triple['articles'][num_articles] = article

                    # We've processed one more article
                    num_articles += 1

            except:

                # Something went wrong
                continue
            
            # Skip further processing of this line if it is not valid
            if not validline:
                continue

            # Get the outlier
            onetwo = list(linecats[0] & linecats[1])
            twothree = list(linecats[1] & linecats[2])
            onethree = list(linecats[0] & linecats[2])

            # Is there one set with more overlap?
            if len(onetwo) == len(twothree) == len(onethree): 
                continue

            # which one is the outlier?
            outlier = 0

            if len(onetwo) > len(twothree) and len(onetwo) > len(onethree):
                outlier = 2

            if len(twothree) > len(onetwo) and len(twothree) > len(onethree):
                outlier = 1

            # Store outlier
            triple['outlier'] = outlier

            # Save it
            pickle.dump(triple, open('../data/wiki/triples/' + str(processed) + '.p', 'wb'))

            # We've processed one more line
            processed += 1

            # Print
            print("Stored triplet " + str(processed) + " [" + str(iterated) + "]")

            # Break for testing purposes
            # if processed > 50:
            #     break


if __name__ == '__main__':
    main()
