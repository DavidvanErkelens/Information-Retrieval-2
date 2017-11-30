# Imports
import urllib.request
import pypandoc
import lxml.etree as ET
from bs4 import BeautifulSoup
import re

# Main function
def main():

    # Open file with Wikipedia triplets, downloaded from http://cs.stanford.edu/ ̃quocle/triplets-data.tar.gz
    with open('wikipedia_2014_09_27_examples.txt') as file:
        
        # Read contents of the line
        contents = file.readlines()

        # Number of processed triplets
        processed = 0

        # Loop over lines
        for line in contents:

            # Skip commented lines
            if line[0] == '#':
                continue

            # Number of processed articles
            article = 0

            # Save categories per article
            linecats = {}

            # Is this line valid (i.e. do all articles exist and are they parsable)?
            validline = True

            # Loop over articles
            for item in line.split():

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
                linecats[article] = set(cats)

                # We've processed one more article
                article += 1

                # Cleanup the last bits of HTML from the plain text
                clean = BeautifulSoup(converted, "lxml").text

                # TODO: Save & tokenize file using existing code
            
            # Skip further processing of this line if it is not valid
            if not validline:
                continue

            # Debug prints comparing categories
            print("1 & 2:")
            print(list(linecats[0] & linecats[1]))
            print("2 & 3:")
            print(list(linecats[1] & linecats[2]))
            print("1 & 3:")
            print(list(linecats[0] & linecats[2]))
            print("\n\n----------\n\n")

            # We've processed one more line
            processed += 1

            # Break for testing purposes
            if processed > 50:
                break


if __name__ == '__main__':
    main()
