
class FeatureExtractor:

    @staticmethod
    def token_features(tagged_sentence, i):
        if (i >= 0) and (i < len(tagged_sentence)):
            return 'w={}'.format(tagged_sentence[i][0])

    @staticmethod
    def previous_tag_feature(tagged_sentence, i):
        if i == 0:
            return 'pos-1=start'
        else:
            return 'pos-1={}'.format(tagged_sentence[i - 1][1])
