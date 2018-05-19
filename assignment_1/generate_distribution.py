from collections import OrderedDict


class GenerateDistribution:
    def __init__(self):
        self.vocab = {}
        self.vocab_prob_dist = {}
        self.con_pos_vocab_prob_dist = {}
        self.con_pos_prob_dist = {}
        self.length_dist = {}

    def build_vocab(self, corpus):
        """
        Build vocabulary from corpus

        Parameters:
            corpus  --  list of list of tuples
        """
        for sent in corpus:
            for token in sent:
                if token[0] not in self.vocab:
                    self.vocab[token[0]] = 1
                else:
                    self.vocab[token[0]] += 1

    def token_prob_dist(self, corpus):
        """
        Computes probability distribution of each token

        Parameters:
            corpus  --  list of list of tuples
        """
        if not self.vocab:
            self.build_vocab(corpus)

        total_freq = sum(self.vocab.values())
        for token, freq in self.vocab.items():
            self.vocab_prob_dist[token] = freq / total_freq

        self.vocab_prob_dist = OrderedDict(sorted(self.vocab_prob_dist.items(),
                                                  key=lambda x: x[1], reverse=True))

    def con_pos_token_prob_dist(self, corpus):
        """
        Computes conditional probability distribution of each token given pos
        { pos: {token: 0.01, ... token: 0.2}, ... pos: {token: 0.4, ... token: 0.011} }

        Parameters:
            corpus  --  list of list of tuples
        """
        for sent in corpus:
            for token, pos in sent:
                if pos not in self.con_pos_vocab_prob_dist:
                    self.con_pos_vocab_prob_dist[pos] = {}

                if token not in self.con_pos_vocab_prob_dist[pos]:
                    self.con_pos_vocab_prob_dist[pos][token] = 1
                else:
                    self.con_pos_vocab_prob_dist[pos][token] += 1

        for k in self.con_pos_vocab_prob_dist:
            total_freq = sum(self.con_pos_vocab_prob_dist[k].values())
            for key, val in self.con_pos_vocab_prob_dist[k].items():
                self.con_pos_vocab_prob_dist[k][key] = val / total_freq

            self.con_pos_vocab_prob_dist[k] = OrderedDict(sorted(self.con_pos_vocab_prob_dist[k].items(),
                                                                 key=lambda x: x[1], reverse=True))

    def pos_prob_dist(self, corpus):
        """
        Computes conditional probability distribution of each pos given previous pos in the sentence
        { prev_pos: {pos: 0.01, ... pos: 0.2}, ... prev_pos: {pos: 0.4, ... pos: 0.011} }

        Parameters:
            corpus  --  list of list of tuples
        """
        for sent in corpus:
            prev_pos = '<pos>'
            for _, pos in sent:
                if prev_pos not in self.con_pos_prob_dist:
                    self.con_pos_prob_dist[prev_pos] = {}

                if pos not in self.con_pos_prob_dist[prev_pos]:
                    self.con_pos_prob_dist[prev_pos][pos] = 1
                else:
                    self.con_pos_prob_dist[prev_pos][pos] += 1
                prev_pos = pos

        for k in self.con_pos_prob_dist:
            total_freq = sum(self.con_pos_prob_dist[k].values())
            for key, val in self.con_pos_prob_dist[k].items():
                self.con_pos_prob_dist[k][key] = val / total_freq

            self.con_pos_prob_dist[k] = OrderedDict(sorted(self.con_pos_prob_dist[k].items(),
                                                           key=lambda x: x[1], reverse=True))

    def length_prob_dist(self, corpus):
        """
        Computes probability distribution of sentence lengths

        Parameters:
            corpus  --  list of list of tuples
        """
        for sent in corpus:
            sent_len = len(sent)
            if sent_len not in self.length_dist:
                self.length_dist[sent_len] = 1
            else:
                self.length_dist[sent_len] += 1

            total_len = sum(self.length_dist.values())
            for k, l in self.length_dist.items():
                self.length_dist[k] = l / total_len

        self.length_dist = OrderedDict(sorted(self.length_dist.items(),
                                              key=lambda x: x[1], reverse=True))
