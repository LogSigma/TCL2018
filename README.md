# TCL2018
Deep Learning 기반 Text Summarization 및 감성분석 모델 설계

## 요구사항
- python 3.6
- konlpy
- nltk
- numpy

## 실행방법
- ### 문장을 형태소 분석하여 NaiveBayesClassifier을 통해 긍정,부정을 검증
  - python [nlp_nltk.py](https://github.com/LogSigma/TCL2018/blob/master/nlp_nltk.py)
- ### RNN을 활용하여 긍정,부정을 분석
  - python [nlp_rnn.py](https://github.com/LogSigma/TCL2018/blob/master/nlp_rnn.py)
- ### CNN를 활용한 긍정,부정 분석
  - python [nlp_cnn.py](https://github.com/LogSigma/TCL2018/blob/master/nlp_cnn.py)
