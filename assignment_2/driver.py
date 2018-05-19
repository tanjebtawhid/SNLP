from assignment_1.tiger_corpus_reader import TigerCorpusReader
from assignment_2.hmm_tagger import HMMTagger
import os
import time


DATA_DIR = 'C:\\Users\\ttanj\\PycharmProjects\\SNLP\\data'
CORPUS_NAME = 'tiger_release_dec05.xml'

reader = TigerCorpusReader()

start = time.time()
reader.read(os.path.join(DATA_DIR, CORPUS_NAME))
tagger = HMMTagger()
# tagger.iterative_train(reader, os.path.join(DATA_DIR, CORPUS_NAME), 50000)
tagger.train(reader.get_sentences(0, 50000))
end = time.time()
print(end - start)