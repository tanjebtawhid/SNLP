from assignment_3.max_ent_tagger import MaxEntTagger

alpha = 0.5
batch_size = 1
epochs = 3

# train = [[('Der', 'ART'), ('Mann', 'NN'), ('ist', 'VBZ'), ('toll', 'ADJ')],
#          [('Die', 'ART'), ('Frau', 'NN'), ('ist', 'VBZ'), ('entzückend', 'ADJ')]]
#
# dev = [[('Die', 'ART'), ('Frau', 'NN'), ('ist', 'VBZ'), ('entzückend', 'ADJ')]]

train = [[('a', 'q'), ('b', 'r')]]
dev = [[('a', 'q'), ('b', 'r')]]

tagger = MaxEntTagger()
tagger.train(train, dev, alpha, batch_size, epochs)
