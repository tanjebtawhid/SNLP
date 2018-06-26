from assignment_1.tiger_corpus_reader import TigerCorpusReader
from assignment_3.max_ent_tagger import MaxEntTagger

import os
import pickle


DATA_DIR = 'C:\\Users\\ttanj\\PycharmProjects\\SNLP\\data'
CORPUS_NAME = 'tiger_release_dec05.xml'

alpha = 0.05
batch_size = 100
epochs = 10

if os.path.exists(os.path.join(DATA_DIR, 'tiger_release_dec05')):
    with open(os.path.join(DATA_DIR, 'tiger_release_dec05'), 'rb') as fh:
        corpus = pickle.load(fh)
        reader = TigerCorpusReader(corpus)
else:
    reader = TigerCorpusReader()
    reader.read(os.path.join(DATA_DIR, CORPUS_NAME))

train = reader.get_sentences(0, 44999)
dev = reader.get_sentences(45000, 45010)

tagger = MaxEntTagger()
tagger.train(train, dev, alpha, batch_size, epochs)
