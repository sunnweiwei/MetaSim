from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

model_corpus = [[['hello', 'you', 'are', 'my']]]
corpus = [['hello', 'you', 'are']]

print(corpus_bleu(model_corpus, corpus, smoothing_function=SmoothingFunction().method1))