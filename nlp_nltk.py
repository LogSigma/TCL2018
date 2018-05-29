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
  
# 데이터 읽기
train_data = read_data('data/ratings_train.txt')
test_data = read_data('data/ratings_test.txt')

# 형태소 분류
train_docs = [(tokenize(row[1]), row[2]) for row in train_data[1:]]
test_docs = [(tokenize(row[1]), row[2]) for row in test_data[1:]]

# Training 데이터 token 수집
tokens = [t for d in train_docs for t in d[0]]
text = nltk.Text(tokens, name='NMSC')

# trem의 따른 분류
selected_words = [f[0] for f in text.vocab().most_common(2000)]
train_xy = [(term_exists(d), c) for d, c in train_docs]
test_xy = [(term_exists(d), c) for d, c in test_docs]

# nltk의 NaiveBayesClassifier 결과 확인
classifier = nltk.NaiveBayesClassifier.train(train_xy)
print(nltk.classify.accuracy(classifier, test_xy))

# 가장 높은 특징 10개 
classifier.show_most_informative_features(10)
