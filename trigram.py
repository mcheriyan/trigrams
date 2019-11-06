import random
import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk import bigrams, trigrams
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter, defaultdict

#create a folder for your corpus
corpusdir = 'miscme/'
newcorpus = PlaintextCorpusReader(corpusdir, '.*')
#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#tokenizer.tokenize(newcorpus.strip())
words = newcorpus.words()
sents = newcorpus.sents()

words = [w.lower() for w in words]
sents = [[w.lower() for w in sent] for sent in sents]

trigram_counts = defaultdict(lambda: Counter())

for sentence in sents:
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        trigram_counts[(w1, w2)][w3] += 1

trigram_probs = defaultdict(lambda: Counter())
for w1_w2 in trigram_counts:
    total_count = float(sum(trigram_counts[w1_w2].values()))
    trigram_probs[w1_w2] = Counter({w3: c/total_count for w3,c in trigram_counts[w1_w2].items()})

for i in range(10):

    text = [None, None] # You can put your own first two words in here

    sentence_finished = False

    # Generate words until two consecutive Nones are generated
    while not sentence_finished:
        r = random.random()
        accumulator = .0
        latest_bigram = tuple(text[-2:])
        prob_dist = trigram_probs[latest_bigram] # prob dist of what token comes next

        for word,p in prob_dist.items():
            accumulator += p
            if accumulator >= r:
                text.append(word)
                break

        if text[-2:] == [None, None]:
            sentence_finished = True


    print(' '.join([t for t in text if t]))
