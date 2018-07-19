import csv
from assignment_4.lda_computation import LDAComputation

words = [['the', 'lecturer', 'love', 'all', 'students'],
         ['Croatia', 'will', 'win', 'the', 'soccer', 'worldcup'],
         ['students', 'and', 'lecturer', 'love', 'watching', 'soccer']]

# with open('C:\\Users\\ttanj\\PycharmProjects\\SNLP\\data\\sentiment_data.csv', 'r', encoding='utf8') as fh:
#     reader = csv.reader(fh, delimiter='\t')
#     data = []
#     for row in reader:
#         data.append(row[1].lower().split())

lda = LDAComputation(num_topics=2, words=words, alpha=1, beta=1)
lda.sample(iteration=10)
lda.print_word_topic_dist()
lda.print_doc_topic_dist()
print()
lda.print_topic_assignments()
