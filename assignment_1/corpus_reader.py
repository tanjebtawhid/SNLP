from abc import ABC, abstractmethod


class CorpusReader(ABC):

    @abstractmethod
    def read(self, file_path):
        raise NotImplementedError

    @abstractmethod
    def get_iterator(self):
        raise NotImplementedError

    @abstractmethod
    def size(self):
        raise NotImplementedError

    @abstractmethod
    def get_sentences(self):
        raise NotImplementedError
