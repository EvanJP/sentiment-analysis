from __future__ import division

from sets import Set
import pandas as pd
import numpy as np 
#import skflow #can be installed via pip install git+git://github.com/tensorflow/skflow.git
from sklearn import datasets, metrics, preprocessing
from sklearn.naive_bayes import GaussianNB

def expand_chars(text, characters):
	for char in characters:
		text = text.replace(char, " "+char+" ")
	return text

def split_text(text):
	score = int(text[-1])
	text = expand_chars(text[:text.index('\t')], '".,()[]{}:;').split(" ")
	text_clean = [cell for cell in text if len(cell)>1]
	text_lc = [cell.lower() for cell in text_clean]
	return text_lc, score

bag_df = pd.DataFrame(0, columns=[0, 1], index=[])
pos_words = []
neg_words = []
words_set = Set()
imdb_labeled = np.genfromtxt('imdb_labelled.txt', delimiter='\n', dtype=None)
amaz_labeled = np.genfromtxt('amazon_cells_labelled.txt', delimiter='\n', dtype=None)
yelp_labeled = np.genfromtxt('yelp_labelled.txt', delimiter='\n', dtype=None)
reviews = [imdb_labeled, amaz_labeled, yelp_labeled]

for company in reviews:
	for review in company:
		text, score = split_text(review)
		for word in text:
			if word not in words_set:
				words_set.add(word)
				bag_df.loc[word] = [0, 0]
				bag_df.ix[word][score] += 1
			else:
				bag_df.ix[word][score] += 1

for index, row in bag_df.iterrows():
	col_0_sc = int(row[0])
	col_1_sc = int(row[1])
	total = col_0_sc + col_1_sc
	if (total) > 2: #ensures few words aren't added
		if(col_0_sc/total) > 0.3 or (col_1_sc/total) > 0.3: #ratio is important to ensure words such as "the" or "and" aren't included
			if(col_0_sc > col_1_sc):
				neg_words.append(index)
			else:
				pos_words.append(index)

training_labels = np.concatenate((pos_words, neg_words))
training_scores = np.concatenate((np.full(len(pos_words), 1), np.full(len(neg_words), 0)))

#required to change strings to labels
label_encode = preprocessing.LabelEncoder()
label_encode.fit(training_labels)
numerical_labels = label_encode.transform(training_labels)
classifier = GaussianNB()
classifier.fit(numerical_labels.reshape(len(numerical_labels), 1), training_scores)

test_labels = []
for review in amaz_labeled:
	text, score = split_text(review)
	test_labels.append([text, score])

total_score = 0
final_score = 0
failed_score = 0
for review in test_labels:
	score = 0
	test = False
	for word in review[0]:
		try:
			label = label_encode.transform(word)
			predi = classifier.predict(label)
			if predi==1:
				score += 1
			else:
				score -= 1
		except ValueError:
			pass
	if score > 0:
		score = 1
		final_score += 1
		test = True
	elif score == 0:
		failed_score += 1
	else:
		score = 0
		final_score += 1
		test = True
	if test:
		if score == review[1]:
			total_score += 1

print 'Final Percentage = ', (total_score/final_score) * 100, '%'
print 'Failed Labeling = ', (failed_score/len(test_labels)) * 100, '%'