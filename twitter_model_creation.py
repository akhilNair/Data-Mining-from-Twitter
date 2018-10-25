from pymongo import MongoClient
'''
Author      : Akhil Nair
Created on  : 4/10/18
Modified on :
'''


import csv
import pandas as pd
import time
import fuzzy_lookup
import nltk
import re
import string
from copy import deepcopy
from nltk.corpus import stopwords
from nltk import sent_tokenize
import utils_twitter
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold
import pickle
import lookup_new
'''------------------------------------------------------------------------
		 MONGODB CONFIGURATION  
''' ###------------------------------------------------------------------------

try:
	connection = MongoClient()
	print('Connection successful')
	#create a db
	db = connection.database
	#create a collection
	collection = db.twitter_db
	
except:
	print('Connection failed')

'''
#import filtered data from L2
df_train = pd.read_csv('Output/L2_filtered_tweet_json_clone_trial.csv',sep = '~',usecols = ['l1_text','REACTION'])
'''
#df_train = pd.read_csv('Output/ConsolidatedTrainFinalFile.csv',sep = ',',usecols = #['l1_text','REACTION'])
#df_train.to_csv('Data/event_train_file_twitter.csv')


def fFeatures(sent):
    return sent.split(',')
	
def fWrite_output_to_file(tweet_id):		
	with open('Output/tweet_extraction_ID.csv','w') as file:
		writer = csv.writer(file)
		for index,tweet in enumerate(tweets_by_id):
			output = [tweet_id[index],tweet.text]	
			#print(output)
			writer.writerows([output])
				

def read_mongo(db,collection):
	cursor = db['twitter_db'].find()
	#expand the cursor and construct the dataframe
	tweet_df = pd.DataFrame(list(cursor))
	#print(tweet_df.head())
	return tweet_df

def fHandleMask(tweet,handle_list):
	masked_tweet = deepcopy(tweet)
	#print('Tweet before masking',tweet)
	for handle in handle_list:
		#masked_tweet = re.sub(handle,'*HANDLE*',masked_tweet)
		masked_tweet = re.sub(handle,'',masked_tweet)
	#print('Masked tweet :',tweet)
	
	return masked_tweet
	
def fMaskTweet(tweet):
	handle_list = re.findall(r'[@]\S+',tweet)
	masked_tweet = fHandleMask(tweet,handle_list)
	return masked_tweet
	
def fPreProcessing(train_df):
	masked_tweet_list = []
	
	for index,tweet in enumerate(list(train_df['TWEET'])):
		url_list = []

		#convert tweet to standard encoding format
		#tweet = tweet.decode('utf-8').encode('ascii','ignore')
		#print(tweet)
		#extract hyperlinks from tweet
		hyperlink_list = re.finditer(r'(http[s]?:\/\/)[a-zA-Z0-9\.\/]+[\.com]+',tweet)
		for hyperlink in (hyperlink_list):
			#print(hyperlink.group())
			url_list.append(hyperlink.group())
			tweet = re.sub(hyperlink.group(),' ',tweet)
			
		print('original tweet',tweet)
		masked_tweet = fMaskTweet(tweet)
		
		#convert tweet into lower case
		masked_tweet = masked_tweet.lower()
		
		'''
		#remove stop words
		word_list = masked_tweet.lower().split()
		stopword_list = set(stopwords.words('english'))
		filtered_word_list = [word for word in word_list if word not in stopword_list]
		masked_tweet = ' '.join(filtered_word_list)
		'''
		
		#filter only alphabetical characters
		#tweet = re.sub(r'[^a-zA-Z\.]',' ',tweet)
		masked_tweet = re.sub(r'[(^b\')]','',masked_tweet)
		
		#remove retweet mentions
		masked_tweet = re.sub(r'(rt)*','',masked_tweet)
		print('masked tweet',masked_tweet)
		
		
		#print('filtered tweet : ',tweet)
		#function to mask handle names in tweet		
		masked_tweet_list.append([masked_tweet,url_list])
	#print(train_df.shape,len(masked_tweet_list))
	train_df['MASKED'] = masked_tweet_list
	return train_df
	
def fUpdateTrainDF(df_train):
	#print('Train DF Shape :',df_train.shape)
	#print(df_train.head())
	
	masked_text_list = []
	event_list = []
	for index,text in enumerate(list(df_train['l1_text'])):
		
		#print('Count :',index)
		#print('Original TWEET :',text)
		text,url_list = utils_twitter.fSentProcessing(text)
		
		#print('Filttered text :',text)
		
		#tmpReactionDict = utils_twitter.fLookupReaction(text)
		#df_reaction = pd.DataFrame(tmpReactionDict,columns = ['REACTION','MASKED'])
		
		'''
		for key in df_reaction:
			#print('KEY :',key)
			detail_list = []
			if key == 'REACTION':
				value = list(df_reaction.get(key).keys())
				reactions = ','.join(list(value))
				print('REACTIONS :',reactions)
				event_list.append(reactions)
				masked_text_list.append(text)
		print('*'*10)	
		'''
		masked_text_list.append(text)
		#df_train['REACTION'] = event_list
	df_train['MASKED'] = masked_text_list
	
	return df_train
	
def fCreateFeatures(sent):
	word_list = nltk.word_tokenize(sent)
	sent_lemmatized_list = [lemmatizer.lemmatize(word) for word in word_list]
	uni_features = ['uni_'+word for word in word_list]
	bi_features = []
	for index,word in enumerate(word_list):
		if index < len(word_list) -2 :
			bi_features.append('bi_'+word_list[index]+'_'+word_list[index+1])
	features = uni_features + bi_features
	features = ','.join(features)
	print('Features :',features)
	#unigrams = ngrams(sent_lemmatized_list,1)
	
	return features
	
def fFeatureExtraction(df_train,_logger):
	##### *IMP* *IMP* PLEASE SET THE INDEX FOR TEXT COLUMN IN DATAFRAME
	#text_index = 7
	text_index = 9
	drug_index =11
	
	
	#df_train = df_train.dropna()
	df_train.dropna(subset = ['REACTION'])
	reaction_list = df_train['REACTION']
	finalReactionList = []
	finalIndicationList = []
	masked_sent_list = []
	symptom_list = []
	#output_list = []
	text_list = []
	feature_list = []
	status_list = []
	non_medical_terms_list = ['k','me','all','hot','got','na+','intolerant','k+','af','tem',
						'married','high','p+','paracetamol','laughter','criminal','condom',
						'suffering','pcm','interaction','eng','se++','invalid','la','benefit unexpected','chemotherapy','ca++','p+','orphan','br-','heavy smoker','lead','ph','nmr','pressure','tension','caffeine','confused','hangover','nai','para','ms','overdose','top','marriage','induction','mg++','disease progression','ra']
	for index,reaction_row in enumerate(reaction_list):
		reactions = reaction_row.split(',')	
		reactions_updated = [reaction for reaction in reactions if reaction.lower() not in non_medical_terms_list]
		drugs = df_train.iloc[index,drug_index]
		print('Drugs :',drugs)
		indicationList,reactionList = lookup_new.check_if_indication(reactions_updated,drugs)
		print('indication list :',indicationList)
		print('Reaction list :',reactionList)
		for reaction in reactions_updated:
			#_logger.info("Current Reaction :=%s",str(reaction))
			#print('Reaction search :',reaction,df_train.iloc[index,3])
			sent = df_train.iloc[index,text_index]	
			print('SENT :',sent)
			exp = re.escape(reaction.lower())+'\D*'
			masked_sent = re.sub(exp,'*SYMP*',sent)
			#_logger.info("Masked Sentence:=%s",masked_sent)
			#print('masked sent',masked_sent)
			sent_list = sent_tokenize(masked_sent)
			for sent in sent_list:
				if '*SYMP*' in sent:
					masked_sent_list.append(sent)
					text_list.append(df_train.iloc[index,text_index])
					symptom_list.append(reaction)
					
												
					#features = fCreateFeatures(sent)
					#feature_list.append(features)
					'''
					tfidf,model = pickle.load(open('Model/Twitter_model.pkl','rb'))
					X_test = tfidf.transform([features])
					prediction = model.predict(X_test)
					if prediction[0] == 1:
						print('Adverse event found ',reaction)
						_logger.info("*** CLASSIFIER PASSED ***")
						_logger.info("Identified Adverse Event ---> = %s ",str(reaction))
						reactionList.append(reaction)
					else:
						_logger.info("*** CLASSIFIER FAILED ***")
						_logger.info("Failed to classify Adverse Event ---> = %s ",str(reaction))
						print('Not an Adverse event :',reaction)
					'''
		if len(reactionList) > 0 or len(indicationList) > 0:
			status_list.append('PASS')
		else:
			status_list.append('FAIL')
	
		if len(reactionList) > 0:	
			extractedReactions = ','.join(reactionList)
		else:
			extractedReactions = ' '
			
		if len(indicationList) > 0:	
			extractedIndications = ','.join(indicationList)
		else:
			extractedIndications = ' '
		
		finalReactionList.append(extractedReactions)	
		finalIndicationList.append(extractedIndications)
		
	#df_training_features = pd.DataFrame({'MASKED':masked_sent_list,'SYMPTOM':symptom_list,'FEATURES' : feature_list})
	df_train['Extracted_AE'] = finalReactionList
	df_train['Extracted_Indication'] = finalIndicationList
	l1_status_list = ['PASS'] * len(finalReactionList)
	l2_status_list = ['PASS'] * len(finalReactionList)
	df_train['l1_status'] = l1_status_list
	df_train['l2_status'] = l2_status_list
	df_train['l3_status'] = status_list
	ae_status_list = ['PRESENT' if status == 'PASS' else 'NOT PRESENT' for status in status_list ]
	df_train['ae_status'] = ae_status_list
	print('*'*10)
	df_train.to_csv('Testing/Output/l3_tweet_json_final.csv',sep = '~')
	#_logger.info("L3 EXTRACTION COMPLETED")
	return df_train
	
def fModelCreate(df_train_features):
	features = df_train_features['FEATURES']
	y = df_train_features['LABEL']
	svmModel = svm.SVC(kernel='rbf', C=10, gamma=0.01)
	tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, tokenizer=fFeatures)
	X = tfidf.fit_transform(features)
	print(X.shape,y.shape)	
	svmModel.fit(X, y)
	
	cv = KFold(n_splits=5)
	score = cross_val_score(svmModel, X, y, cv=cv)
	print('Final Score is:',score)
	print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
	
	filename = 'Model/Twitter_model_final.pkl'
	
	pickle.dump((tfidf, svmModel), open(filename, 'wb'))
	print('Model Created !')
	
def fExtractFromTweet(sent):
	finalReactionList = []
	sent,url_list = utils_twitter.fSentProcessing(sent)
	tmpReactionDict = utils_twitter.fLookupReaction(sent)
	print('Reaction dict :',tmpReactionDict)
	for reaction in tmpReactionDict.get('REACTION').keys():
		print('Check for reaction : ',reaction)
		exp = re.escape(reaction.lower())+'\D*'
		masked_sent = re.sub(exp,'*SYMP*',sent)
		#print('masked sent',masked_sent)
		sent_list = sent_tokenize(masked_sent)
		for sent in sent_list:
			if '*SYMP*' in sent:
				features = fCreateFeatures(sent)
				
				tfidf,model = pickle.load(open('Model/Twitter_model.pkl','rb'))
				features = fCreateFeatures(tweet)
				X_test = tfidf.transform([features])
				prediction = model.predict(X_test)
				if prediction[0] == 1:
					print('Adverse event found :',reaction)
					finalReactionList.append(reaction)
				else:
					print('Not an Adverse event :',reaction)
				
	print('Final Reaction list :',finalReactionList)
'''
if __name__ == '__main__':
	#df_trial = df_train.iloc[:,:]
	print('YOLO')
	tweet = '@suresh i have been experiencing headache and pain.'
	#df_train_updated = fUpdateTrainDF(df_train)
	#df_train_updated = df_train
	#print('#'*10)
	#df_train_updated = pd.read_csv('Data/event_train_file_twitter_updated_22_10.csv',sep = '~')
	#df_train_updated = pd.read_csv('Data/event_train_file_twitter_updated_tylenol.csv',sep = '~')
	#print(df_train.head())
	#print('#'*10)
	#df_train_features = fFeatureExtraction(df_train_updated)
	#print(df_train.head())
	df_train_features = pd.read_csv('Data/features_train_file_final.csv',sep = '~')
	fModelCreate(df_train_features)
	#train_df = fPreProcessing(train_df)
	#rain_df.to_csv('train_file.csv')
	
	#fExtractFromTweet(tweet)
	
	#tmp function to create features
	sent = 'i just asked if anyone had tylenol bc i have a *SYMP* and this girl gave me a pill that was not tylenol but i sti'
	features = fCreateFeatures(sent)
	print('Extrcted features :',features)
'''

	