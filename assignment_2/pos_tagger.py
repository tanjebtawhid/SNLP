from abc import ABC, abstractmethod


class POSTagger(ABC):

    @abstractmethod
    def train(self, corpus):
        raise NotImplementedError

    @abstractmethod
    def predict(self, sentence):
        raise NotImplementedError
