import nltk
import pandas as pd
from konlpy.tag import Twitter

twitter = Twitter()
feature_count=2000

# 1. 

# 데이터 읽기
train_data = pd.read_scv('data/ratings_train.txt', sep='\t', encoding='CP949')
test_data = pd.read_scv('data/ratings_test.txt', sep='\t', encoding='CP949')

# 형태소 분류
train_data = [(['/'.join(t) for t in twitter.pos(row[2], norm=True, stem=True)], row[3]) for row in train_data.itertuples()]
test_data = [(['/'.join(t) for t in twitter.pos(row[2], norm=True, stem=True)], row[3]) for row in test_data.itertuples()]

# Training 데이터 token 수집
tokens = [t for d in train_data for t in d[0]]

text = nltk.Text(tokens, name='NMSC')
print(text.vocab().most_common(10))

# trem의 따른 분류
selected_words = [f[0] for f in text.vocab().most_common(feature_count)]
train_xy = [({'exists({})'.format(word): (word in set(d)) for word in selected_words}, c) for d, c in train_data]
test_xy = [({'exists({})'.format(word): (word in set(d)) for word in selected_words}, c) for d, c in test_data]

# nltk의 NaiveBayesClassifier 결과 확인
classifier = nltk.NaiveBayesClassifier.train(train_xy)
print(nltk.classify.accuracy(classifier, test_xy))

# 가장 높은 특징 10개 확인
classifier.show_most_informative_features(10)
