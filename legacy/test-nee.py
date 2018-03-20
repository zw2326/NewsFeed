# Named entity extraction from NLTK. Works well.
# Also consider terminology extraction: https://pypi.python.org/pypi/topia.termextract/1.1.0
# Named entity extraction with ML:
# http://nlpforhackers.io/named-entity-extraction/
# http://nlpforhackers.io/training-ner-large-dataset/

import nltk
import code
sample = input(">> ")

sentences = nltk.sent_tokenize(sample)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
print(tagged_sentences)
chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
print(type(chunked_sentences))

def extract_entity_names(t):
    print(type(t))
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names

entity_names = []
for tree in chunked_sentences:
    # Print results per sentence
    # print extract_entity_names(tree)

    entity_names.extend(extract_entity_names(tree))

# Print all entity names
#print entity_names

# Print unique entity names
print(set(entity_names))