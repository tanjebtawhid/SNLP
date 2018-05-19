from assignment_1.tiger_corpus_reader import TigerCorpusReader
from assignment_1.generate_distribution import GenerateDistribution
import numpy as np


def sample(dist_dict):
    """
    Samples token/pos given probability distribution

    Parameters:
        dist_dict  --  dict, probability distribution

    Returns:+-
        item  --  string, token/pos
    """
    x = np.random.uniform(0, 1)
    running_sum = 0

    for item, prob in dist_dict.items():
        running_sum += prob
        if running_sum - x >= 0:
            return item


def naive_text_generator(obj):
    """
    Generates sentence for a sampled length

    Parameters:
        obj  --  GenerateDistribution, instance
    Returns:
        sent  --  string, generated sentence
    """
    sent_len = sample(obj.length_dist)
    sent = ''
    for i in range(sent_len):
        sent += sample(obj.vocab_prob_dist) + ' '
    return sent


def text_generator(obj):
    """
    Generates sentence for a sampled length using
    conditional probability distribution

    Parameters:
        obj  --  GenerateDistribution, instance
    Returns:
        sent  --  string, generated sentence
    """
    sent_len = sample(obj.length_dist)
    sent = ''

    prev_pos = sample(obj.con_pos_prob_dist['<pos>'])
    token = sample(obj.con_pos_vocab_prob_dist[prev_pos])
    sent += token + ' '
    for i in range(sent_len - 1):
        pos = sample(obj.con_pos_prob_dist[prev_pos])
        sent += sample(obj.con_pos_vocab_prob_dist[pos]) + ' '
        prev_pos = pos
    return sent


corpus = TigerCorpusReader()
corpus.read('tiger_release_dec05.xml')
print(corpus.size())
# print(corpus.get_sentences(10, 11))

dist = GenerateDistribution()
dist.token_prob_dist(corpus.corpus)
dist.length_prob_dist(corpus.corpus)

for i in range(10):
    print(naive_text_generator(dist))
print()

dist.pos_prob_dist(corpus.corpus)
dist.con_pos_token_prob_dist(corpus.corpus)

for i in range(10):
    print(text_generator(dist))


print(sum(dist.con_pos_vocab_prob_dist['NN'].values()))
print(sum(dist.con_pos_prob_dist['<pos>'].values()))
