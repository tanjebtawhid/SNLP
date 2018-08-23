
class TaggerEvaluator:
    def __init__(self):
        self.tag_gold_freq = {}
        self.tag_pred_freq = {}
        self.tag_pred_correct = {}
        self.accuracy = None

    def get_statistics(self):
        """
        computes precision and recall for each tag

        Returns:
            string  --  formatted string
        """
        string = ''
        string += 'Accuracy: {:f}\n'.format(self.accuracy)
        string += '#tags: {:d}\n'.format(len(self.tag_gold_freq))

        for tag in self.tag_pred_freq:
            if tag in self.tag_pred_correct:
                string += 'Precision({}): {}/{}\n'.format(tag, self.tag_pred_correct[tag], self.tag_pred_freq[tag])
            else:
                string += 'Precision({}): {}/{}\n'.format(tag, 0, self.tag_pred_freq[tag])

        for tag in self.tag_gold_freq:
            if tag in self.tag_pred_correct:
                string += 'Recall({}): {}/{}\n'.format(tag, self.tag_pred_correct[tag], self.tag_gold_freq[tag])
            else:
                string += 'Recall({}): {}/{}\n'.format(tag, 0, self.tag_gold_freq[tag])

        return string

    def evaluate(self, tagger, tagged_sentences):
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
            sent = [each[0] for each in sentence]
            print('Evaluating on sentence: {}'.format(sentence))

            prediction = tagger.predict(sent)
            print('Prediction: {}'.format(prediction))

            for j in range(len(sentence)):
                total += 1
                target_tag = sentence[j][1]
                predicted_tag = prediction[j][1]
                if target_tag == predicted_tag:
                    correct += 1
                    self.update(self.tag_pred_correct, predicted_tag)

                self.update(self.tag_gold_freq, target_tag)
                self.update(self.tag_pred_freq, predicted_tag)

            self.accuracy = correct / total

    def update(self, d, tag):
        """
        Updates predictions statistics

        Parameters:
            d    --  dictionary
            tag  --  string, pos
        Returns:
        """
        if tag in d:
            d[tag] += 1
        else:
            d[tag] = 1
