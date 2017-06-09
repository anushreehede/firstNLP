import nltk.corpus.util
from nltk.classify import MaxentClassifier
from nltk.corpus import opinion_lexicon
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def create_feature(words):
	useful_words = [word for word in words if word not in stopwords.words("english")]
	my_dict = dict([(word, True) for word in useful_words])
	return my_dict

# create tuples of negative words
neg_words = []
words = opinion_lexicon.words("negative-words.txt")
neg_words.append((create_feature(words), "Negative"))

# create tuples of positive words
pos_words = []
words = opinion_lexicon.words("positive-words.txt")
pos_words.append((create_feature(words), "Positive"))

train_set = neg_words[:3587] + pos_words[:1504]
test_set = neg_words[-1196:] + pos_words[-501:]

#print(test_set)

algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
classifier = nltk.MaxentClassifier.train(train_set, algorithm, max_iter=100)
#classifier.show_most_informative_features(10)

accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(accuracy * 100)

text = input("Enter some text ")

words = word_tokenize(text)
wordlist = create_feature(words)
# print(wordlist)
# print(type(wordlist))
# print(nltk.classify.util.accuracy(classifier, wordlist))
print(classifier.classify(wordlist))
probs = classifier.prob_classify(wordlist)
# print(probs.samples())
print((probs.prob("Positive")*100),"%", probs.prob("Negative")*100,"%")





