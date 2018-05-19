from assignment_1.tiger_corpus_reader import TigerCorpusReader
from assignment_2.hmm_tagger import HMMTagger
from assignment_2.tagger_evaluator import TaggerEvaluator
import os
import time


DATA_DIR = 'C:\\Users\\ttanj\\PycharmProjects\\SNLP\\data'
CORPUS_NAME = 'tiger_release_dec05.xml'

start = time.time()

reader = TigerCorpusReader()
reader.read(os.path.join(DATA_DIR, CORPUS_NAME))

evaluator = TaggerEvaluator()

train = reader.get_sentences(0, 50000)
tagger = HMMTagger()
tagger.train(train)

test = reader.get_sentences(50000, reader.size())
evaluator.evaluate(tagger, test)
print(evaluator.get_statistics())

end = time.time()
print(end - start)
