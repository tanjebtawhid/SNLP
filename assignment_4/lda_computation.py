from typing import List
import numpy as np


class LDAComputation:
    def __init__(self, num_topics: int, words: List[List[str]], alpha: int, beta: int):
        """
        Parameters
        ----------
        num_topics : int
            number of topics
        words : List[List[str]]
            corpus
        alpha : int
        beta : int
        """
        self.num_topics = num_topics
        self.words = words
        self.alpha = alpha
        self.beta = beta

        self.__topics = []  # list of docs, each doc is list of topic assigned to each word
        self.__doc_topic_count = {}  # topic count for each document {doc: {topic: count}}
        self.__topic_word_count = {}  # word count for each topic {topic: {word: count}}
        self.__topic_count = {}  # topic count over corpus

        self.__num_words = len(set([word for doc in self.words for word in doc]))

    def sample(self, iteration):
        """
        Parameters
        ----------
        iteration : int
            Number of times to iterate over the corpus
        """
        self.__initialize_topics()
        self.print_topic_assignments()
        for t in range(iteration):
            # print('Iteration:{}'.format(t + 1))
            for i, doc in enumerate(self.words):
                for j, word in enumerate(doc):
                    topic = self.__topics[i][j]
                    self.__decrease_count(topic, word, i)
                    topic_dist = np.asarray([self.__compute_topic_dist(k, word, i) for k in range(self.num_topics)])
                    new_topic = int(np.random.choice(self.num_topics, 1, p=topic_dist / sum(topic_dist)))
                    # new_topic = int(np.argmax(topic_dist))
                    self.__update_count(new_topic, word, i)
                    self.__topics[i][j] = new_topic

    def __initialize_topics(self):
        """Randomly assign each word of each document to a topic and
        campute topic wise vocabulary and dccument wise topic distributions
        """
        for i, doc in enumerate(self.words):
            topic_list = []
            for word in doc:
                sampled_topic = int(np.random.choice(self.num_topics, 1))
                topic_list.append(sampled_topic)
                self.__update_count(sampled_topic, word, i)
            self.__topics.append(topic_list)

    def print_topic_assignments(self):
        for i, doc in enumerate(self.words):
            print('Document {}'.format(i), end=': ')
            for j, word in enumerate(doc):
                print('{}/{}'.format(word, self.__topics[i][j]), end=', ')
            print()

    def print_word_topic_dist(self):
        print('\nTopic wise word distribution...')
        for i in range(self.num_topics):
            print('Topic:', i)
            total = sum(self.__topic_word_count[i].values())
            if total is not 0:
                for word in self.__topic_word_count[i]:
                    print('{} -> {:.4f}'.format(word, self.__topic_word_count[i][word] / total), end='\t')
                print()

    def print_doc_topic_dist(self):
        print('\nDocument wise topic distribution...')
        for i in range(len(self.words)):
            print('Document:', i)
            total = sum(self.__doc_topic_count[i].values())
            for topic in self.__doc_topic_count[i]:
                print('{} -> {:.4f}'.format(topic, self.__doc_topic_count[i][topic] / total), end='\t')
            print()

    def __get_topic_word_count(self, topic: int, word: str) -> int:
        """Returns number of times word w appear in topic k
        """
        if word in self.__topic_word_count[topic]:
            return self.__topic_word_count[topic][word]
        else:
            return int(0)

    def __get_doc_topic_count(self, doc, topic) -> int:
        """Returns number of times topic k appear in document d
        """
        if topic in self.__doc_topic_count[doc]:
            return self.__doc_topic_count[doc][topic]
        else:
            return int(0)

    def __decrease_count(self, topic: int, word: str, doc: int):
        """Decreases all relevant counts associated with the topic
        of word w in document d
        """
        self.__doc_topic_count[doc][topic] -= 1
        self.__topic_word_count[topic][word] -= 1
        self.__topic_count[topic] -= 1

    def __update_count(self, topic, word, doc):
        """Updates topic count, topic wise vocabulary count
        and document wise topic count
        Parameters
        ----------
        topic : int
            index of topic
        word : str
            word/token from dth document
        doc : int
            index of document
        """
        self.__update_topic(topic)
        self.__update_word_topic(word, topic)
        self.__update_document_topic(doc, topic)

    def __compute_topic_dist(self, topic: int, word: str, doc: int) -> float:
        """Computes likelihood of topic k in document d.
        """
        doc_lh = self.__get_doc_topic_count(doc, topic) + self.alpha
        word_lh = (self.__get_topic_word_count(topic, word) + self.beta) / \
                  (sum(self.__topic_word_count[topic].values()) + (self.__num_words * self.beta))
        return doc_lh * word_lh

    def __update_topic(self, topic: int):
        """Updates corpus wide topic count
        Parameters
        ----------
        topic : int
            topic index
        """
        if topic in self.__topic_count:
            self.__topic_count[topic] += 1
        else:
            self.__topic_count[topic] = 1

    def __update_word_topic(self, word: str, topic: int):
        """Updates topic wise vocabulary count
        Parameters
        ----------
        word : str
            current word
        topic : int
            topic index
        """
        if topic in self.__topic_word_count:
            if word in self.__topic_word_count[topic]:
                self.__topic_word_count[topic][word] += 1
            else:
                self.__topic_word_count[topic][word] = 1
        else:
            self.__topic_word_count[topic] = {}
            self.__topic_word_count[topic][word] = 1

    def __update_document_topic(self, doc: int, topic: int):
        """Updates document wise topic count
        Parameters
        ----------
        doc : int
            index of document
        topic : int
            index of topic
        """
        if doc in self.__doc_topic_count:
            if topic in self.__doc_topic_count[doc]:
                self.__doc_topic_count[doc][topic] += 1
            else:
                self.__doc_topic_count[doc][topic] = 1
        else:
            self.__doc_topic_count[doc] = {}
            self.__doc_topic_count[doc][topic] = 1
