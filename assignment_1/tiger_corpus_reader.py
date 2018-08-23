import re
import pickle
import os

from assignment_1.corpus_reader import CorpusReader


class TigerCorpusReader(CorpusReader):
    """
    Corpus is represented as a list of lists, where each inner list is a sentence. Each sentence consists of
    tuples containing (token, pos) pair.

    corpus -- [ [(token_1, pos_1)...(token_n, pos_n)], ... [(token_1, pos_1)...(token_n, pos_n)] ]
    """
    def __init__(self, corpus=None):
        """
        Initialize with an empty corpus
        """
        if corpus is None:
            self.corpus = []
        else:
            self.corpus = corpus

    def read(self, file_path):
        """
        Reads corpus into self.corpus

        Parameters:
            file_path  --  path to corpus
        """
        regex = re.compile(r'<t id="(.+)" word="(.+)" lemma="(.+)" pos="(.+)" morph="(.+)"')

        with open(file_path, 'r') as reader:
            for line in reader:
                if re.search(r'\s?<terminals>\n', line):
                    sent = []

                matches = regex.search(line)
                if matches:
                    sent.append((matches.group(2), matches.group(4)))

                if re.search(r'</terminals>', line):
                    self.corpus.append(sent)
                    # yield sent

    def get_iterator(self):
        raise NotImplementedError

    def size(self):
        """
        Returns size of the corpus
        Number of sentences in this case
        """
        return len(self.corpus)

    def get_sentences(self, i, j):
        """
        Returns sentences from index i to j
        """
        # for idx in range(i, j):
        #     yield self.corpus[idx]
        return self.corpus[i:j]

    def to_file(self, directory, fname):
        """
        Writes read corpus into file

        Parameters:
            directory  --  string
            fname      --  string, file name
        """
        with open(os.path.join(directory, fname), 'wb') as fh:
            pickle.dump(self.corpus, fh)
