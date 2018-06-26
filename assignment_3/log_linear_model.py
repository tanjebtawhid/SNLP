from collections import OrderedDict
import numpy as np


class LogLinearModel:

    def __init__(self):
        self.weights = None  # list of floats
        self.feature_index = OrderedDict()  # dictionary of dictionaries { string: {string: int ...} }
        self.int_to_feature = OrderedDict()  # dictionary {int: string}
        self.labels = set()
        self.no_features = 0

    def initialize(self, feature_list, labels):
        """
        Parameters:
            feature_list  --  list of tuples, (token_feature, previous_tag_feature) for each token
            labels        --  list, pos tag each token
        """
        for label in labels:
            self.labels.add(label)

        for features in feature_list:
            for feat in features:
                self.add_feature(feat)

        self.weights = np.ones(self.no_features)

    def add_feature(self, feat):
        for label in self.labels:
            if label not in self.feature_index:
                self.feature_index[label] = OrderedDict()
                self.feature_index[label][feat] = self.no_features
                self.int_to_feature[self.no_features] = '{} tag={}'.format(feat, label)
                self.no_features += 1
            else:
                if feat not in self.feature_index[label]:
                    self.feature_index[label][feat] = self.no_features
                    self.int_to_feature[self.no_features] = '{} tag={}'.format(feat, label)
                    self.no_features += 1

    def contains_feature(self, feat, label):
        # idx = self.get_feature_index(feat, label)
        # if emp_count[idx] > 0:
        #     return True
        # else:
        #     return False
        if label in self.feature_index:
            return feat in self.feature_index[label]
        else:
            return False

    def get_feature_index(self, feat, label):
        if label in self.feature_index:
            if feat in self.feature_index[label]:
                return self.feature_index[label][feat]
            else:
                return None
        else:
            return None

    def update_weights(self, weights):
        self.weights = weights

    def compute_unnormalized_prob(self, features, label):
        prob = 0.
        for feat in features:
            idx = self.get_feature_index(feat, label)
            # print('({}, {}):idx-{}'.format(feat, label, idx))
            if idx is not None:
                prob += self.weights[idx]
                # print('({}, {}):prob-{}'.format(feat, label, prob))

        return np.exp(prob)

    def compute_sum_prob(self, features):
        sum_prob = 0.
        for label in self.labels:
            sum_prob += self.compute_unnormalized_prob(features, label)

        return sum_prob

    def compute_prob(self, features, label):
        return self.compute_unnormalized_prob(features, label) / self.compute_sum_prob(features)

    def log_likelihood(self, feature_list, labels):
        ll = 0.
        for features, label in zip(feature_list, labels):
            ll += np.log(self.compute_prob(features, label))

        return ll
