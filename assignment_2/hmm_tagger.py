from assignment_2.pos_tagger import POSTagger
import numpy as np


class HMMTagger(POSTagger):
    def __init__(self):
        self.state_transition_counts = {}  # dictionary of dictionaries
        self.state_emission_counts = {}  # dictionary of dictionaries

        self.state_transition_total = {}
        self.state_emission_total = {}

        self.state_transitions = {}  # dictionary of dictionaries
        self.state_emissions = {}  # dictionary of dictionaries

        self.pos_index = {}
        self.inv_pos_index = {}

        self.token_freq = {}

    def train(self, sentences):
        """
        Trains HMM pos tagger

        Parameters:
            sentences  --  list of list of tuples
        """
        num_sentences = 0
        for sent in sentences:
            self.update(sent)
            num_sentences += 1

        self.compute_model()
        print('Trained on {} sentences'.format(num_sentences))

    def iterative_train(self, reader, file_path, train_size):
        """
        Trains HMM pos tagger. Gets one sentence at a time

        Parameters:
            reader      --  instance, TigerCorpusReader
            file_path   --  path to corpus
            train_size  --  size of the training set
        """
        num_sentences = 0
        for sent in reader.read(file_path):
            self.update(sent)
            num_sentences += 1
            if num_sentences == train_size:
                break

        self.compute_model()
        print('Trained on {} sentences'.format(num_sentences))

    def compute_model(self):
        """
        Computes state transition and emission probabilities Applies add one smoothing for
        emission probabilities to deal with unseen tokens during training. Cretes po to index
        and index to pas mapping.
        """
        for prev_tag in self.state_transition_counts:
            self.state_transitions[prev_tag] = {}
            total = self.state_transition_total[prev_tag]
            for tag in self.state_transition_counts[prev_tag]:
                self.state_transitions[prev_tag][tag] = self.state_transition_counts[prev_tag][tag] / total

        for tag in self.state_emission_counts:
            self.state_emissions[tag] = {}
            total = self.state_emission_total[tag]
            for token in self.state_emission_counts[tag]:
                self.state_emissions[tag][token] = \
                    (self.state_emission_counts[tag][token] + 1) / (total + len(self.token_freq) + 1)

            # Smoothing
            self.state_emissions[tag]['unkwn'] = 1 / (total + len(self.token_freq) + 1)

        for idx, tag in enumerate(self.state_transitions):
            if tag not in self.pos_index:
                self.pos_index[tag] = idx
                self.inv_pos_index[idx] = tag

    def update(self, sentence):
        """
        Given a sentence from training set Updates state transition,
        state emission and token frequency count

        Parameters:
            sentence  --  list of tuples [ (token, tag) ... (token, tag) ]
        """
        for i in range(len(sentence)):
            token = sentence[i][0]
            if i == 0:
                prev_tag = '<pos>'
            else:
                prev_tag = sentence[i - 1][1]

            tag = sentence[i][1]

            self.update_state_transitions(prev_tag, tag)
            self.update_emissions(tag, token)
            self.update_token_freq(token)

    def update_token_freq(self, token):
        """
        Updates token frequency count

        Parameters:
            token  --  string, word
        """
        if token in self.token_freq:
            self.token_freq[token] += 1
        else:
            self.token_freq[token] = 1

    def update_emissions(self, tag, token):
        """
        Updates emission probabiltry for the given tag and token pair

        Parameters:
            tag    --  string, pos
            token  --  string, word
        """
        if tag in self.state_emission_counts:
            if token in self.state_emission_counts[tag]:
                self.state_emission_counts[tag][token] += 1
            else:
                self.state_emission_counts[tag][token] = 1
        else:
            self.state_emission_counts[tag] = {}
            self.state_emission_counts[tag][token] = 1

        if tag in self.state_emission_total:
            self.state_emission_total[tag] += 1
        else:
            self.state_emission_total[tag] = 1

    def update_state_transitions(self, prev_tag, tag):
        if prev_tag in self.state_transition_counts:
            if tag in self.state_transition_counts[prev_tag]:
                self.state_transition_counts[prev_tag][tag] += 1
            else:
                self.state_transition_counts[prev_tag][tag] = 1
        else:
            self.state_transition_counts[prev_tag] = {}
            self.state_transition_counts[prev_tag][tag] = 1

        if prev_tag in self.state_transition_total:
            self.state_transition_total[prev_tag] += 1
        else:
            self.state_transition_total[prev_tag] = 1

    def predict(self, sentence):
        """
        Predict most likely tag sequence

        Parameters:
            sentence  --  list, [token1, token2, ... ]
        Returns:
            Most likely tag sequence for the sentence
        """
        return self.viterbi(sentence)

    def viterbi(self, sentence):
        """
        A sentence is a list of tuples. Here our goal is to infer the most likely tag sequence
        for the given sentence, therefore, input sentence will be just list of tokens

        Parameters:
            sentence  --  list
        Returns:
            tagged_sentence  --  list of tuples, [ (token, tag) ... (token, tag) ]
        """
        delta = np.zeros(shape=(len(self.state_transitions), len(sentence)), dtype=float)
        tagged_sentence = []

        for next_tag in self.state_transitions['<pos>']:
            if next_tag in self.state_transitions['<pos>']:
                a = self.a('<pos>', next_tag)
            else:
                a = 0.

            if sentence[0] not in self.state_emissions[next_tag]:
                b = self.b(next_tag, 'unkwn')
            else:
                b = self.b(next_tag, sentence[0])

            idx = self.pos_index[next_tag]
            delta[idx, 0] = a * b

        ml_tag = self.inv_pos_index[np.argmax(delta[:, 0])]  # holds most likely tag at the specific time step
        tagged_sentence.append((sentence[0], ml_tag))

        for j in range(1, len(sentence)):
            for next_tag in self.state_transitions:
                if next_tag != '<pos>':
                    next_tag_idx = self.pos_index[next_tag]
                    if sentence[j] not in self.state_emissions[next_tag]:
                        b = self.b(next_tag, 'unkwn')
                    else:
                        b = self.b(next_tag, sentence[j])

                    for tag in self.state_transitions[next_tag]:
                        if next_tag in self.state_transitions[tag]:
                            a = self.a(tag, next_tag)
                        else:
                            a = 0.
                        tag_idx = self.pos_index[tag]
                        temp = delta[tag_idx, j - 1] * a * b
                        if temp > delta[next_tag_idx, j]:
                            delta[next_tag_idx, j] = temp

            ml_tag = self.inv_pos_index[np.argmax(delta[:, j])]
            tagged_sentence.append((sentence[j], ml_tag))

        return tagged_sentence

    def print_array(self, delta, i):
        """
        Parameters:
            delta  --  2D array of float
            i      --  integer
        """
        print('{}: '.format(i))
        for k in range(delta.shape[1]):
            if delta[i, k] > np.NINF:
                print('{:s}:{:f} '.format(self.inv_pos_index[k], delta[i, k]))
        print()

    def delta(self, s, j, delta):
        """
        Parameters:
            s      --  string
            i      --  integer
            delta  --  dictionary of dictionaries { int: {string: float} ...}
        Returns:
        """
        if j in delta:
            if s in delta[j]:
                return float(delta[s])
        return 0.

    def b(self, tag, token):
        """
        Returns probabilty of emitting the token given the tag

        Parameters:
            tag    --  string, pos
            token  --  string, word
        Returns:
            emission probability
        """
        return self.state_emissions[tag][token]

    def a(self, tag, next_tag):
        """
        Returns transitions probabilty from tag to next_tag

        Parameters:
            tag       --  string, pos
            next_tag  --  string, pos
        Returns:
            state transition probability
        """
        return self.state_transitions[tag][next_tag]

    def to_string(self):
        """
        Forms a formated string containing all state transition and
        emission probabilities

        Returns:
            string  -- string
        """
        string = ''
        string += '{:s}\n'.format('=' * 50)
        string += 'State transition probabilities for {:d} tags:\n'.format(len(self.state_transitions))
        for prev_tag in self.state_transitions:
            string += '{:s}: '.format(prev_tag)
            for tag in self.state_transitions[prev_tag]:
                string += '{:s}({:f}) '.format(tag, self.state_transitions[prev_tag][tag])
            string += '\n'

        string += '{:s}\n'.format('=' * 50)
        string += 'State emission probabilities for {:d} tokens:\n'.format(len(self.state_emissions))
        for tag in self.state_emissions:
            string += '{:s}: '.format(tag)
            for token in self.state_emissions[tag]:
                string += '{:s}({:f}) '.format(token, self.state_emissions[tag][token])
            string += '\n'

        return string
