from pymongo import MongoClient
'''
Author      : Akhil Nair
Created on  : 4/10/18
Modified on :
'''

import tweepy
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
import lookup_new 
import utils_twitter
import logs
import twitter_model_creation


'''------------------------------------------------------------------------
		 TWITTER API CONFIGURATION  
''' ###------------------------------------------------------------------------
 
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

# Authorization to consumer key and consumer secret 
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

# Access to user's access key and access secret 
auth.set_access_token(access_token, access_token_secret) 

# Calling api 
api = tweepy.API(auth)



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

_logger = logs.init_log('Logs/Test/Twitter.log')

'''------------------------------------------------------------------------
		 Dictionary Import 
''' ###------------------------------------------------------------------------

#drug dictionary to extract tweets from world wide web
df_drug_dictionary = pd.read_csv('Dictionary/dictionary_drug_tweet.csv')
df_drug_dictionary = df_drug_dictionary['DRUG']
print(df_drug_dictionary.head())
			
			
def fFeatures(sent):
    return sent.split(',')
	
def fFilterDataFrame(df_l1_data):
	print('Filtering started')
	df_filtered = df_l1_data.dropna(subset = ['REACTION'])
	return df_filtered
	
#function to filter	extracted data based on certain boolean sequences
def fApplyF2Filter(df_l1_data):
	_logger.info("******************************")
	_logger.info("L2 Extraction Started")
	final_e2b = []
	tweet_text_list = df_l1_data['l1_text']
	drug_list = []
	event_list = []
	url_list = []
	
	### extract country names from the location data present in meta information
	location_list = df_l1_data['l1_location']
	country_list = [lookup_new.country_elookup(str(location)) for location in location_list]
	df_l1_data['l1_location'] = country_list
	for index,tweet_text in enumerate(list(tweet_text_list)):
		finalDict,embedded_url = fPreProcessing(str(tweet_text))
		url_list.append(embedded_url)
		df_final_dict = pd.DataFrame(finalDict,columns = ['DRUG','REACTION'])
		#print(finalDict)
		for key in df_final_dict:
			value = list(finalDict.get(key).keys())
			detail_list = []
			if key == 'DRUG':
				drugs = ','.join(list(value))
				print('DRUGS :',drugs)
				drug_list.append(drugs)
			elif key == 'REACTION':
				reactions = ','.join(list(value))
				print('REACTIONS :',reactions)
				event_list.append(reactions)
			print('*'*10)
		_logger.info("Drugs extracted from loookup :=%s",str(drugs))
		_logger.info("Reactions extracted from loookup :=%s",str(reactions))
	
	if len(event_list) > 0:
		df_l1_data['DRUG'] = drug_list
		df_l1_data['REACTION'] = event_list
		df_l1_data['EMBEDDED_URL'] = url_list
		
		df_l1_data.to_csv('Testing/Output/L2_tweet_json_final.csv',sep = '~')
		df_l1_data_filtered = fFilterDataFrame(df_l1_data)
	
		df_l1_data_filtered.to_csv('Testing/Output/L2_filtered_tweet_final.csv',sep = '~')
		_logger.info("L2 Extraction completed :")
		return df_l1_data_filtered
	else:
		_logger.info("No Reactions extracted from loookup for the dataset ")
		return []
	


def text_elookup_dict(v_CIOMS_text, x_fd=[], v_filter_list=None):
	"""
	Input is a free form text. Exact Matching Lookup on dictionary and fetching other values in the row of the dictionary.
	"""
	
	isEvent = False

	regex = re.compile('[%s]' % re.escape(string.punctuation.replace('/', '')))

	# initialize reaction dict
	reaction = dict()

	# to add matched reaction and its start index in text
	reaction_indices_dict = dict()

	# convert input text to lower case
	v_CIOMS_text_lower = v_CIOMS_text.lower()

	# Tokenizer needed to have words without punctuation marks
	w_input_list_lower = nltk.word_tokenize(v_CIOMS_text_lower)
	input_lower_punc = regex.sub(' ', v_CIOMS_text_lower)

	# loop over all reactions, sorted by length of reactions (largest reaction will come first)
	for w in sorted(x_fd, key=len, reverse=True):
		# lower, remove punctuations, lower
		w_lower = w.lower()
		w_punc = regex.sub(' ', w)
		w_lower_punc = regex.sub(' ', w_lower)

		# if current reaction in text
		if w_lower_punc in input_lower_punc:
			matched_string = ''
			# match exact word
			pattern_w_lower_punc = r'\b' + w_lower_punc + r'\b'
			matched = re.search(pattern_w_lower_punc, input_lower_punc)
			if matched:
				p = matched.start(0)
				matched_string = matched.group(0)

			# if not then check if current reaction of only one token then
			# check if this token in token list of input text
			elif len(nltk.word_tokenize(w)) == 1:
				if w_lower in w_input_list_lower:  # special case - to handle shortforms of <6 characters
					p = v_CIOMS_text.find(w)
					matched_string = w

			# else take out current reaction from text
			else:
				p = v_CIOMS_text_lower.find(w_lower)
				matched_string = v_CIOMS_text[
								 v_CIOMS_text_lower.find(w_lower):v_CIOMS_text_lower.find(w_lower) + len(w_lower)]

			# if matched reaction is not none then add it in reaction dict
			if matched_string != '':

				# check if current reaction is not part of any of the previously identified reaction
				# using indices of current reaction and previously identified reactions
				reaction_part_added = False
				for key, start_index in reaction_indices_dict.items():
					if start_index <= p <= start_index + len(key):
						reaction_part_added = True

				# if current reaction is not part of any previously identified reactions then add
				if reaction_part_added is False:
					isEvent = True
				else:
					isEvent = False
	return isEvent

def fUpdateFinalDict(entityDict,finalDict,type = ''):
	'''
	Function to update finalDict and keep unqiue values only
	
	Param value     : Drug or Reaction name returned by lookup_new (STRING) 
	Param finalDict : The global final ictionary which stores PADR values (DICTIONARY)
	Param type 		: Flag to decide between drug and Reaction
	'''
	if type in finalDict:
		finalDetails = finalDict[type]
		for entityValue,entityData in entityDict.items():
			if len([key for key in finalDetails if key.lower() == entityValue.lower()]) > 0:
				return finalDict
			else:
				finalDetails[entityValue] = entityData
		finalDict[type] = finalDetails
	else:
		finalDict[type] = entityDict
	return finalDict
			
			
def fLookupDrug(masked_sent,finalDict):
	#print('sent for drug :',masked_sent)
	tmpDrugDict = lookup_new.text_flookup_drug(masked_sent)
	#print('Drug Dict: ',tmpDrugDict)
	#function to update finalDict to keep unique elements
	finalDict = fUpdateFinalDict(tmpDrugDict,finalDict,'DRUG')
	return finalDict
	

def fLookupReaction(masked_sent,finalDict = {}):
	tmpReactionDict = lookup_new.text_flookup_dict(masked_sent)
	#print('Reaction dict :',tmpReactionDict)
	#function to update finalDict to keep unique elements
	finalDict = fUpdateFinalDict(tmpReactionDict,finalDict,'REACTION')
	return finalDict
	
#function to transform relevant details meta information into relevant json fields
def fExtractInfoFromTweet(twt_satus_list):
	structured_tweet_list = []
	structured_tweet_list_with_meta = []
	for index,twt in enumerate(twt_satus_list):
		print('Processing tweet No. :',index)
		_json_details = twt._json
		text = _json_details['text'].encode('ascii','ignore')
		created_date = _json_details['created_at']
		user_id = twt.user.id_str
		tweet_id = twt.id
		reporter = twt.user.name
		location = lookup_new.country_elookup(twt.user.location)
		source = twt.source
		structured_tweet_list_with_meta.append({'l1_content':twt})
		structured_tweet_list.append({'l1_tweet_id':tweet_id,'user_id':user_id,'l1_patient':reporter,'l1_reporter':reporter,'l1_location':location,'l1_source':source,'l1_text' : text,'l1_creation_date':created_date})
	df_l1_data_with_meta = pd.DataFrame(structured_tweet_list_with_meta,columns = ['l1_content'])
	df_l1_data_str = pd.DataFrame(structured_tweet_list,columns = ['l1_tweet_id','user_id','l1_patient','l1_reporter','l1_location','l1_source','l1_text','l1_creation_date'])
	df_l1_data_with_meta.to_csv('Output/L1_meta_info_integrated.csv')
	df_l1_data_str.to_csv('Output/L1_tweet_json_clone_integrated.csv',sep ='~')
	
	_logger.info("L1 Extraction Completed")
	return df_l1_data_str
		
#function to extract all tweets with a set of queries from world wide web
def fApplyF1Filter(df_drug_dictionary):
	
	_logger.info("L1 Filtering Started")

	#_logger.error('Error in case : ' + "current_source_document")
	
	#tweets_by_query = api.search(q = twitter_handle)
	#print('Length of query :',len(list(df_drug_dictionary)))
	for query in list(df_drug_dictionary):
		print('Query String :',query)
		df_l1_data = {}
		max_tweets = 100
		try:
			#search by query string
			twt_satus_list = list(tweepy.Cursor(api.search, q=query,lang = 'en').items())
			df_l1_data = fExtractInfoFromTweet(twt_satus_list)
			#collection.insert_many(twt)
		except tweepy.TweepError  as e:
			print(e)
			_logger.error('Tweepy Error : URL Limit Exceeded')
			time.sleep(60 * 15)
		except StopIteration:
			break
	return df_l1_data
	
def fPreProcessing(tweet):
	url_list = []
	finalDict = {}
	#print('Tweet :',type(tweet), tweet)
	
	#sentence tokenization
	sents = sent_tokenize(tweet)
	for sent in sents:
		print('Sent :',sent)
		#function to clean sent and separate url
		masked_sent,url_list = utils_twitter.fSentProcessing(sent,url_list)
		_logger.info("URLs Extracted from text :=%s",str(url_list))
		_logger.info("Filtered text :=%s",str(masked_sent))
		print('masked tweet',masked_sent)
		#print('Url :',url_list)
		
		#function to extract drugs from the sentence
		finalDict = fLookupDrug(masked_sent,finalDict)
		
		# function to extract all the possible rections in the sentence
		finalDict = fLookupReaction(masked_sent,finalDict)
	embedded_url = ','.join(url_list)
	print('Final Dict :',finalDict)	
	return finalDict,embedded_url

def fFeatureExtraction(df_l2_data):
	df_train_updated = twitter_model_creation.fUpdateTrainDF(df_l2_data)
	df_train_features = twitter_model_creation.fFeatureExtraction(df_train_updated,_logger)
	
def fApplyF3Filter(df_l2_data):
	#_logger.info("*******************************")
	#_logger.info("L3 EXTRACTION STARTED")
	fFeatureExtraction(df_l2_data)

		
if __name__ == '__main__':
	
	#tweets = ["b'@upasbook Great read as always. I was on paracetamol for 5 days. Cold turkey had #sweats, migraine, tremors while on &amp; 3 days after."]
	#df_l1_data = fApplyF1Filter(df_drug_dictionary)
	
	#df_l1_data = pd.read_csv('Output/L1_tweet_json_clone_22_10_18.csv',sep ='~')
	#filter data based on adr 
	#df_l2_data = fApplyF2Filter(df_l1_data)
	
	#df_l1_data = pd.read_csv('Testing/Output/L2_tweet_json_final.csv',sep = '~')
	#df_l1_data_filtered = fFilterDataFrame(df_l1_data)
	
	#df_l1_data_filtered.to_csv('Testing/Output/L2_filtered_tweet_final.csv',sep = '~')
	#df_l1_data_filtered.to_csv('Output/L2_filtered_tweet_json_clone_trial.csv',sep = '~')
	
	df_l2_data = pd.read_csv('Testing/Output/L2_filtered_tweet_final.csv',sep = '~') 
	df_l2_data_filtered = df_l2_data.dropna(subset = ['DRUG'])
	df_l2_data_filtered.to_csv('Output/L2_drug_filtered_tweet_json_clone_trial.csv',sep = '~')
	if len(df_l2_data) != 0:
		print('L3 STARTING')
		df_l3_data = fApplyF3Filter(df_l2_data_filtered)
	else:
		print('NO CASE')
	
	