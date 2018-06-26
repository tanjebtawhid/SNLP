from abc import ABC, abstractmethod


class POSTagger(ABC):

    @abstractmethod
    def train(self, train, dev, alpha, batch_size, epochs):
        """
        Parameters:
            train       --  train data
            dev         --  validation data
            alpha       --  float, learning rate
            batch_size  --  int
            epochs      --  int
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, sentence):
        """
        Parameters:
            sentence  --  list of tuples, [ (token, pos) ... ]
        Returns:
        """
        raise NotImplementedError
