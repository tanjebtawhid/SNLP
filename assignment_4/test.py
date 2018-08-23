from assignment_4.lda_computation import LDAComputation

words = [
    ['the', 'lecturer', 'love', 'all', 'students'],
    ['Croatia', 'will', 'win', 'the', 'soccer', 'worldcup'],
    ['students', 'and', 'lecturer', 'love', 'watching', 'soccer']
]

lda = LDAComputation(num_topics=4, words=words, alpha=1, beta=1)
lda.sample(iteration=10)
lda.print_word_topic_dist()
lda.print_doc_topic_dist()
print()
lda.print_topic_assignments()
