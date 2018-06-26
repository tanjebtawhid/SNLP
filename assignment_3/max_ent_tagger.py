import numpy as np

from assignment_3.pos_tagger import POSTagger
from assignment_3.feature_extractor import FeatureExtractor
from assignment_3.log_linear_model import LogLinearModel


class MaxEntTagger(POSTagger):

    def __init__(self):
        self.model = LogLinearModel()
        self.labels = []  # list of strings
        self.emperical_feature_count = None

    def predict(self, sentence):
        tagged_sentence = []

        for i, token in enumerate(sentence):
            features = [FeatureExtractor.token_features(sentence, i),
                        FeatureExtractor.previous_tag_feature(sentence, i)]
            maximum = 0.
            for label in self.model.labels:
                prob = self.model.compute_prob(features, label)
                if prob > maximum:
                    maximum = prob
                    max_tag = label

            tagged_sentence.append((token[0], max_tag))

        return tagged_sentence

    def evaluate(self, tagged_sentences):
        """
        Predicts most likey tag sequence for test sentences and record prediction
        statistics, computes prediction accuracy

        Parameters:
            tagger            --  instance, HMMTagger
            tagged_sentences  --  list of list of tuples, sentences with pos tag for each token
        """
        correct = 0
        total = 0

        print('Evaluating on {:d} sentences!\n'.format(len(tagged_sentences)))
        for sentence in tagged_sentences:
            print('Evaluating on sentence: {}'.format(sentence))

            prediction = self.predict(sentence)
            print('Prediction: {}'.format(prediction))

            for j in range(len(sentence)):
                total += 1
                target_tag = sentence[j][1]
                predicted_tag = prediction[j][1]
                if target_tag == predicted_tag:
                    correct += 1
                else:
                    pass

        return correct / total

    def emperical_counts(self, batch):
        emperical_feature_counts = np.zeros(self.model.no_features)

        for sentence in batch:
            for i, token in enumerate(sentence):
                for feat in [FeatureExtractor.token_features(sentence, i),
                             FeatureExtractor.previous_tag_feature(sentence, i)]:
                    if self.model.contains_feature(feat, token[1]):
                        emperical_feature_counts[self.model.get_feature_index(feat, token[1])] += 1

        return emperical_feature_counts

    def model_expectations(self, batch, emp_count):
        expected_counts = np.zeros(self.model.no_features)

        # features_list = []
        # for sentence in batch:
        #     for i, _ in enumerate(sentence):
        #         features_list.append([FeatureExtractor.token_features(sentence, i),
        #                               FeatureExtractor.previous_tag_feature(sentence, i)])
        #
        # for features in features_list:
        #     for label in self.model.labels:
        #         p = self.model.compute_prob(features, label)
        #         for feat in features:
        #             expected_counts[self.model.get_feature_index(feat, label)] += p

        for sentence in batch:
            print('Sentence:{}'.format(sentence))
            for i, _ in enumerate(sentence):
                features = [FeatureExtractor.token_features(sentence, i),
                            FeatureExtractor.previous_tag_feature(sentence, i)]
                print('Features for token {}: {}'.format(i, features))
                for label in self.model.labels:
                    p = self.model.compute_prob(features, label)
                    print('Probabilty of ({}, {}): {}'.format(features, label, p))
                    for feat in features:
                        idx = self.model.get_feature_index(feat, label)
                        print('Feature index of ({}, {}): {}'.format(feat, label, idx))
                        if emp_count[idx] > 0:
                            expected_counts[idx] += p
                            print('Updated expected count of ({}, {}): {}'.format(feat, label, expected_counts[idx]))
                            print(expected_counts)
                        print()
                        # if self.model.contains_feature(feat, label):
                        #     expected_counts[self.model.get_feature_index(feat, label)] += p

        return expected_counts

    def negative_log_likelihood(self, weights, batch):
        self.model.update_weights(weights)
        feature_list = []

        for sentence in batch:
            for i, _ in enumerate(sentence):
                feature_list.append((FeatureExtractor.token_features(sentence, i),
                                     FeatureExtractor.previous_tag_feature(sentence, i)))

            return - self.model.log_likelihood(feature_list, self.labels)

    def train(self, train, dev, alpha, batch_size, epochs):
        no_examples = 0  # number of tokens seen
        feature_list = []  # list of tuples, (token_feature, previous_tag_feature)

        for sentence in train:
            for i, token in enumerate(sentence):
                feature_list.append((FeatureExtractor.token_features(sentence, i),
                                     FeatureExtractor.previous_tag_feature(sentence, i)))
                self.labels.append(token[1])
                no_examples += 1

        print('Train size:{}'.format(len(train)))
        print('Examples:{}'.format(no_examples))
        print('Initializing...')
        self.model.initialize(feature_list, self.labels)
        print('Finished initialization')
        print('Labels:{}'.format(self.model.labels))
        print('Feature index:{}'.format(self.model.feature_index))
        print('int to feature:{}'.format(self.model.int_to_feature))

        # print('Negative log-likelihood:{}'.format(self.model.log_likelihood(feature_list, self.labels)))

        early_stop = False
        no_sentences = 0
        best_score = 0.
        data = []

        k = 1
        while k <= epochs:
            print('Epoch: {}'.format(k))
            for sentence in train:
                data.append(sentence)
                if (no_sentences % batch_size == 0) or (len(train) < batch_size):

                    emp_count = self.emperical_counts(data)
                    print('Emperical count:', end='')
                    for i, val in enumerate(emp_count):
                        print('{}: {}, '.format(self.model.int_to_feature[i], val), end='')
                    print()

                    exp_count = self.model_expectations(data, emp_count)
                    print('Expected count:', end='')
                    for i, val in enumerate(exp_count):
                        print('{}: {}, '.format(self.model.int_to_feature[i], val), end='')
                    print()

                    gradient = emp_count - exp_count
                    print('Gradient:{}'.format(gradient))
                    self.model.weights = self.model.weights + alpha * gradient
                    print('Weights:{}'.format(self.model.weights))
                    data = []
                no_sentences += 1

            # batch_idx = 0
            # if batch_idx <= len(train) - batch_size:
            #     batch = train[batch_idx:batch_idx + batch_size]
            #     emp_count = self.emperical_counts(batch)
            #     print('Emperical count:', end='')
            #     for i, val in enumerate(emp_count):
            #         print('{}: {}, '.format(self.model.int_to_feature[i], val), end='')
            #     print()
            #
            #     exp_count = self.model_expectations(batch, emp_count)
            #     print('Expected count:', end='')
            #     for i, val in enumerate(exp_count):
            #         print('{}: {}, '.format(self.model.int_to_feature[i], val), end='')
            #     print()
            #
            #     gradient = emp_count - exp_count
            #     print('Gradient:{}'.format(gradient))
            #     self.model.weights = self.model.weights + alpha * gradient
            #     print('Weights:{}'.format(self.model.weights))
            #     batch_idx += batch_size

            accuracy = self.evaluate(dev)
            print('Accuracy on dev: {}'.format(accuracy))
            print('Negative log-likelihood: {}'.format(- self.model.log_likelihood(feature_list, self.labels)))

            # if accuracy > best_score:
            #     best_score = accuracy
            # else:
            #     print('Early stopping!!!')
            #     early_stop = True

            k += 1
