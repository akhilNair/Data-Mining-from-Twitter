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

def fHandleMask(tweet,handle_list):
	masked_tweet = deepcopy(tweet)
	#print('Tweet before masking',tweet)
	for handle in handle_list:
		#masked_tweet = re.sub(handle,'*HANDLE*',masked_tweet)
		handle = re.escape(handle.lower())
		masked_tweet = re.sub(handle,'',masked_tweet)
	return masked_tweet
	
def fMaskSent(sent):
	handle_list = re.findall(r'(@\S+\:*)',sent)
	masked_sent = fHandleMask(sent,handle_list)
	return masked_sent,handle_list
	

def fExtractHyperlink(sent,url_list):
	url_list.extend(url_list)
	hyperlink_list = re.finditer(r'(http[s]?:\/\/)[a-zA-Z0-9\.\/]+',sent)
	for hyperlink in (hyperlink_list):
		#print(hyperlink.group())
		url_list.append(hyperlink.group())
		sent = re.sub(hyperlink.group(),' ',sent)
	return sent,url_list
	
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
	
def fLookupReaction(masked_sent,finalDict = {}):
	finalDict = {}
	tmpReactionDict = lookup_new.text_flookup_dict(masked_sent)
	#print('Reaction dict :',tmpReactionDict)
	#function to update finalDict to keep unique elements
	finalDict = fUpdateFinalDict(tmpReactionDict,finalDict,'REACTION')
	return finalDict


#def fExtractSlang(masked_sent):
	
def fSentProcessing(sentence,url_list = []):
	url_list = []
	sent = deepcopy(sentence)
	
	#convert tweet to standard encoding format
	#tweet = tweet.decode('utf-8').encode('ascii','ignore')

	#extract hyperlinks from tweet
	sent,url_list = fExtractHyperlink(sent,url_list)
	
	masked_sent,handle_list = fMaskSent(sent)
	
	#convert tweet into lower case
	masked_sent = masked_sent.lower()
	
	'''
	#remove stop words
	word_list = masked_tweet.lower().split()
	stopword_list = set(stopwords.words('english'))
	filtered_word_list = [word for word in word_list if word not in stopword_list]
	masked_tweet = ' '.join(filtered_word_list)
	'''
	
	#filter only alphabetical characters
	#tweet = re.sub(r'[^a-zA-Z\.]',' ',tweet)
	masked_sent = re.sub(r'(^b[\'\"])','',masked_sent)
	
	#remove retweet mentions
	masked_sent = re.sub(r'(^rt)','',masked_sent)
	
	masked_sent = re.sub(r'[#\?\d+\-\'\,\(\)]*','',masked_sent)
	masked_sent = re.sub(r'\\n','',masked_sent)
	
	#masked_sent = fExtractSlang(masked_sent)
	
	#print('Final filtered sentence :',masked_sent)
	return masked_sent,url_list
