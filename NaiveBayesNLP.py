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
  
