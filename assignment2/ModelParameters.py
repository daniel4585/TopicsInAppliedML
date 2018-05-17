import numpy as np
from collections import Counter
class ModelParameters(object):
    def __init__(self, hyperParams):
        self.hyperParams = hyperParams


    def Init(self, train):
        self.vocabulary = Counter(sum(train, []))
        for i,(key, value) in enumerate(self.vocabulary.items()):
            self.vocabulary[key] = (value, i)
        self.totalNumOfWords = 0
        for word in self.vocabulary.values():
            self.totalNumOfWords += float(word[0])
        self.u = np.random.normal(scale=0.01, loc=0.0, size=(len(self.vocabulary), self.hyperParams.D))
        self.v = np.random.normal(scale=0.01, loc=0.0, size=(len(self.vocabulary), self.hyperParams.D))
        self.normalize()
        denominator = sum([pow(x[0], self.hyperParams.alpha) for x in self.vocabulary.values()])

        self.unigramDistVec = [pow(x[0], self.hyperParams.alpha) / denominator for x in self.vocabulary.values()]
        self.sentenceDistVec = [float(len(x)) / self.totalNumOfWords for x in train]

    def normalize(self):
        for i in range(len(self.vocabulary)):
            self.u[i] = self.u[i] / np.linalg.norm(self.u[i])
            self.v[i] = self.v[i] / np.linalg.norm(self.v[i])

    def sample_word(self):
        return np.random.choice(self.vocabulary.keys(), p=self.unigramDistVec)

    def sample_K_words(self):
        return [self.vocabulary[self.sample_word()][1] for _ in range(self.hyperParams.K)]

    def sample_target_context(self, train):
        chosenSentence = np.random.choice(train, p=self.sentenceDistVec)
        wordIndex = np.random.choice(range(len(chosenSentence)))
        context = [self.vocabulary[chosenSentence[i]][1] for i in range(max(0,wordIndex-self.hyperParams.C), wordIndex)] + \
                  [self.vocabulary[chosenSentence[i]][1] for i in range(wordIndex+1, min(wordIndex + 1 + self.hyperParams.C, len(chosenSentence)))]
        selectedWordVocabularyIndex = self.vocabulary[chosenSentence[wordIndex]][1]
        return selectedWordVocabularyIndex, context



