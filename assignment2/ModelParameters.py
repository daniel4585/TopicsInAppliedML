import numpy as np
from collections import Counter


class ModelParameters(object):
    def __init__(self, hyperParams):
        self.hyperParams = hyperParams
        self.vocabulary = {}
        self.totalNumOfWords = 0
        self.u = None
        self.v = None
        self.unigramDistVec = []
        self.sentenceDistVec = []

    def Init(self, data):
        train = data.train
        test = data.test
        self.vocabulary = Counter(sum(test, []))
        for i, (key, value) in enumerate(self.vocabulary.items()):
            self.vocabulary[key] = 0

        self.vocabulary.update(Counter(sum(train, [])))
        for i, (key, value) in enumerate(self.vocabulary.items()):
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

        self.largest_sentence = 0
        for s in data.train:
            self.largest_sentence = max(self.largest_sentence, len(s))

    def normalize(self):
        for i in range(len(self.vocabulary)):
            self.u[i] = self.u[i] / np.linalg.norm(self.u[i])
            self.v[i] = self.v[i] / np.linalg.norm(self.v[i])

    def sample_word(self):
        return np.random.choice(self.vocabulary.keys(), p=self.unigramDistVec)

    def sample_K_words(self):
         return [self.vocabulary[x][1] for x in np.random.choice(self.vocabulary.keys(), size=self.hyperParams.K, p=self.unigramDistVec)]

    def sample_batch_words(self, batchsize):
        sampled = np.random.choice(self.vocabulary.keys(), size=(batchsize, self.hyperParams.K), p=self.unigramDistVec)
        samples = []
        for i, sample in enumerate(sampled):
            samples.append([self.vocabulary[x][1] for x in sample])
        return samples



    def sample_target_context(self, train ):
        chosenSentence = np.random.choice(train, p=self.sentenceDistVec)
        wordIndex = np.random.choice(range(len(chosenSentence)))
        selectedWordVocabularyIndex, context = self.get_vocabularyIndexAndContext(chosenSentence, wordIndex)
        return selectedWordVocabularyIndex, context

    def sample_multi_target_context(self, train, numOfTargets):
        chosenSentences = np.random.choice(train, p=self.sentenceDistVec, size=(numOfTargets, 1))
        ret = []
        for i, sentence in enumerate(chosenSentences):
            selectedWordVocabularyIndex, context = self.get_vocabularyIndexAndContext(sentence[0], np.random.choice(range(len(sentence))))
            ret.append((selectedWordVocabularyIndex, context))
        return ret

    def get_vocabularyIndexAndContext(self, chosenSentence, wordIndex):
        context = [self.vocabulary[chosenSentence[i]][1] for i in
                   range(max(0, wordIndex - self.hyperParams.C), wordIndex)] + \
                  [self.vocabulary[chosenSentence[i]][1] for i in
                   range(wordIndex + 1, min(wordIndex + 1 + self.hyperParams.C, len(chosenSentence)))]
        selectedWordVocabularyIndex = self.vocabulary[chosenSentence[wordIndex]][1]
        return selectedWordVocabularyIndex, context
