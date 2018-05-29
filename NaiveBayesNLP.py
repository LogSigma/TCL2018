from konlpy.tag import Twitter
import nltk

twitter = Twitter()

def read_data(filename):
  with open(filename, 'r') as f:
    data = [line.split('\t') for line in f.read().splitlines()]
  return data

def tokenize(doc):
  return ['/'.join(t) for t in twitter.pos(doc, norm=True, stem=True)]

def term_exists(doc):
  return {'exists({})'.format(word): (word in set(doc)) for word in selected_words}
  
train_data = read_data('data/ratings_train.txt')
test_data = read_data('data/ratings_test.txt')

##
train_docs = [(tokenize(row[1]), row[2]) for row in train_data[1:]]
test_docs = [(tokenize(row[1]), row[2]) for row in test_data[1:]]

tokens = [t for d in train_docs for t in d[0]]

text = nltk.Text(tokens, name='NMSC')

selected_words = [f[0] for f in text.vocab().most_common(2000)]
train_docs = train_docs[:10000]
train_xy = [(term_exists(d), c) for d, c in train_docs]
test_xy = [(term_exists(d), c) for d, c in test_docs]

classifier = nltk.NaiveBayesClassifier.train(train_xy)
print(nltk.classify.accuracy(classifier, test_xy))

classifier.show_most_informative_features(10)
