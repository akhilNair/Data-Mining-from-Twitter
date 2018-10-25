import csv
import os
import re
import string
import nltk
import sys
import fuzzy_lookup
from collections import OrderedDict
from fuzzywuzzy import fuzz

##########################################################
# Building the list = (LLT_NAME, others) from MedDRA file
with open(os.path.dirname(os.path.realpath(__file__))+ '/' +'dictionary/dictionary_MedDRA.csv') as f_MedDRA:
	next(f_MedDRA, None)  # Ignoring the header
	r = csv.reader(f_MedDRA, delimiter=',', quotechar='"')
	d_list = list(r)
f_MedDRA.close()



def filter_dict(v_filter_list=None):
	"""
	Creating a filter on the necessary rows of the dictionary
	"""

	fd = {}
	# input argument is blank i.e. no filter.
	if v_filter_list is None or len(v_filter_list) == 0:
		for x in sorted((v for v in d_list if v[8] == 'Y'), key=lambda d_list: d_list[0]):
			fd[x[0]] = x[1:11]
	else:
		for x in sorted((v for v in d_list if v[8] == 'Y' and v[9] in v_filter_list), key=lambda d_list: d_list[0]):
			fd[x[0]] = x[1:11]
	return fd


####################################################
# Building the list from IME (Important Medical Event MedDRA version 19.1 from European Medicines Agency) file
with open(os.path.dirname(os.path.realpath(__file__))+ '/' +'dictionary/dictionary_IME.csv', encoding='utf-8', errors='ignore') as f_ime:
	next(f_ime, None)  # Ignoring the header
	r = csv.reader(f_ime, delimiter=',', quotechar='"')
	i_list = list(r)
f_ime.close()

def ime_elookup(v_input=''):
	serious_code = '2'  # 1=Yes, 2=No
	if any(v_input == row[0] for row in i_list) and v_input != '':
		serious_code = '1'
	return serious_code

	
####################################################
# Building the list from Local Company's Drug Dictionary File
# (Generic name, common name, Drug Code, concentration, unit, formulation, country)
with open(os.path.dirname(os.path.realpath(__file__))+ '/' +'dictionary/dictionary_Drug_Local_with_unknown.csv') as f_drug_local:
	next(f_drug_local, None)  # Ignoring the header
	r = csv.reader(f_drug_local, delimiter=',', quotechar='"')
	product_list_local = list(r)
	product_list_local = [drug for drug in product_list_local if len(drug[1]) > 4]
f_drug_local.close()

# Building the list from Combined Drug Dictionary (combined dictionary from WHODD, orange book and RXNORM) file
# (Generic name, common name, Drug Code)
with open(os.path.dirname(os.path.realpath(__file__))+ '/' +'dictionary/dictionary_drug_big_one_sorted.csv') as f_drug_local:
	next(f_drug_local, None)  # Ignoring the header
	r = csv.reader(f_drug_local, delimiter=',', quotechar='"')
	product_list_combined = list(r)
	product_list_combined = [drug for drug in product_list_combined if len(drug[1]) > 4]
	# remove records which are common terms
	if len (product_list_combined) > 0 and len(product_list_combined[0]) > 2:
		product_list_combined = [drug for drug in product_list_combined if drug[2].strip() == '']
f_drug_local.close()


def get_product_code(input_product, product_details_list, v_concentration='', v_unit='', v_formulation='',v_country=''):
	"""
	get codes of input products list based on concentration, unit, formulation and country
	"""

	product_code = ''
	try:
		# input product details list is not None
		if len(product_details_list) > 0:

			# filter rows having product name equal to input product
			filtered_product_list = list(filter(lambda x: x[7].strip() == input_product.strip(), product_details_list))

			# code for current drug is not identified till yet
			code_identified = False

			# for each row in filtered product list
			# compare other fields with given input values
			for row in filtered_product_list:
				if (len(row)) > 2:
					'''
					concentration = row[3]
					unit = row[4]
					formulation = row[5]
					country = row[6]
					'''
					concentration_matched = False
					unit_matched = False
					formulation_matched = False
					country_matched = False

					# match current concentration with input concentration
					if v_concentration == '':
						concentration_matched = True
					elif v_concentration.lower() == row[3].lower():
						concentration_matched = True

					# match current unit with input unit
					if v_unit == '':
						unit_matched = True
					elif v_unit.lower() == row[4].lower():
						unit_matched = True

					# match current formulation with input formulation
					if v_formulation == '':
						formulation_matched = True
					elif v_formulation.lower() == row[5].lower():
						formulation_matched = True

					# match current country with input country
					if v_country == '':
						country_matched = True
					elif v_country.lower() == row[6].lower():
						country_matched = True

					# if all four details are matched then use it's product code
					if concentration_matched and unit_matched and formulation_matched and country_matched:
						product_code = row[2]
						code_identified = True
						break

			# code was not found then take first match and use its product code
			if code_identified is False:
				for row in filtered_product_list:
					if len(row) > 2:
						product_code = row[2]
						break
	except Exception as ex:
		print(ex)

	return product_code

	
	
def text_flookup_drug_from_dict(v_input='', drug_list=None, v_concentration='', v_unit='', v_formulation='', v_country=''):
	"""
	Input is a freeform text after exact match. Fuzzy Matching Lookup using on dictionary and fetching other values
	in the row of the dictionary.
	"""

	# initialize new dict
	drug_dict = OrderedDict()

	# dic to add identified drug and its start index in input text
	drug_indices_dict = dict()

	# convert input text to upper case
	v_input_upper = v_input.upper().strip()

	if v_input_upper != '':

		# remove punctuations and tokenize input string
		regex = re.compile('[%s]' % re.escape(string.punctuation.replace('/', '')))
		# v_input_list = nltk.word_tokenize(regex.sub(' ', v_input_upper))
		v_input_list, v_input_span_list = get_spans(regex.sub(' ', v_input_upper))

		# look for each drug in input text
		for row in drug_list:
			#print(str(row).encode('ascii','ignore'),len(row))
			if len(row) > 3:
				current_drug_name = row[7]
			else:
				current_drug_name = row[1]

			if current_drug_name not in drug_dict.keys():
				word_matched_count = 0
				drug_start_index = sys.maxsize
				drug_end_index = 0
				current_drug_name = current_drug_name.replace(' + ', '+').strip()
				current_drug_name = current_drug_name.replace(' / ', '/').strip()
				current_drug_name = current_drug_name.replace(' /', '/').strip()
				current_drug_name = current_drug_name.replace('/', '/').strip()
				current_drug_name = current_drug_name.replace('&', 'and').strip()
				current_drug_name = current_drug_name.upper().strip()
				drug_tokens = nltk.word_tokenize(current_drug_name)
				for current_token in drug_tokens:

					# current drug token in input text
					if current_token in v_input_list:

						# get start and end index of current token in input text
						index = v_input_list.index(current_token)
						start_index = v_input_span_list[index][1]  # v_input_upper.find(current_token)
						end_index = v_input_span_list[index][2]  # start_index + len(current_token)

						# update current drug start and end index in input text
						if start_index < drug_start_index:
							drug_start_index = start_index
						if end_index > drug_end_index:
							drug_end_index = end_index
						word_matched_count += 1

				if word_matched_count == len(drug_tokens) and (drug_end_index - drug_start_index) < len(
						current_drug_name) + 5:
					drug_part_added = False

					# check if current drug is part of any previously identified drugs
					# using indices of current drug and previously identified drugs
					for key, start_index in drug_indices_dict.items():
						if start_index <= drug_start_index <= start_index + len(key):
							drug_part_added = True

					# if current drug is not part of any previously identified drug then add
					if drug_part_added is False:
						# If drug found in string then
						f_score = fuzzy_lookup.extract_best_match(v_input_upper[drug_start_index:drug_end_index],
																  current_drug_name,
																  scorer=fuzzy_lookup.similarity_using_set)

						# get product code
						# code = get_product_code(row[1], drug_list, v_concentration, v_unit, v_formulation, v_country)
						code = '0000'

						# add generic name and local code in dict
						drug_dict[row[1]] = [row[0], code, f_score[1]]

						# add current drug and its start index
						drug_indices_dict[row[1]] = drug_start_index
	return drug_dict

def modify_drug_sting(v_input):
	"""
	modify drug string
	"""
	try:
		v_input = v_input.lower()
		v_input = re.sub(r'\,$', '', v_input).strip()
		v_input = v_input.replace(' + ', '+').strip()
		v_input = v_input.replace(' / ', '/').strip()
		v_input = v_input.replace(' /', '/').strip()
		v_input = v_input.replace('/', '/').strip()
		v_input = v_input.replace('dr.', 'dr').strip()
		v_input = v_input.replace('scholl\'s', 'scholls').strip()
		v_input = v_input.replace('dr. scholl\'s', 'dr scholls').strip()
		v_input = v_input.replace('&', 'and').strip()
		v_input = v_input.replace('scholl\'s', 'scholls').strip()
		v_input = v_input.replace('12hr', '12 hour').strip()
		v_input = v_input.replace('24hr', '24 hour').strip()
		v_input = re.sub(r'\borig\b', 'original', v_input).strip()
		v_input = re.sub(r'aleve\s+caplets', 'aleve caplet', v_input).strip()
		v_input = re.sub(r'aleve\s+tablets', 'aleve tablet', v_input).strip()
		v_input = v_input.replace('alka-seltzer lemon-lime', 'alka-seltzer lemon lime').strip()
	except Exception as ex:
		logging.exception("message")

	return v_input

	
def text_flookup_drug(v_input='', v_concentration='', v_unit='', v_formulation='', v_country=''):
	"""
	get Drugs from dictionaries using flookup
	"""

	extracted_product_list = []
	ordered_product_dict = OrderedDict()

	# modify drug name
	v_input = modify_drug_sting(v_input)
	print('@1',v_input)
	
	# get product from Local Product List
	product_dict = text_flookup_drug_from_dict(v_input, product_list_local,
											   v_concentration=v_concentration, v_unit=v_unit,
											   v_formulation=v_formulation, v_country=v_country)
	print('@2',product_dict)
	'''
	# if only one product found then search product in local dict using KNN model
	if len(product_dict) <= 1:
		product_dict = search_product_using_knn_model(v_input, v_concentration='', v_unit='', v_formulation='',
													  v_country='')
	'''
	# append details from product dict to main dict
	for current_product_name in product_dict.keys():
		extracted_product_list.append(current_product_name.lower())
		ordered_product_dict[current_product_name] = product_dict.get(current_product_name)

	# get drug from Combined Product list
	additional_product_dict = text_flookup_drug_from_dict(v_input, product_list_combined,
														  v_concentration=v_concentration,
														  v_unit=v_unit, v_formulation=v_formulation,
														  v_country=v_country)

	# check if product identified is not part of products identified using Local dictionary
	for current_product_name, product_details in additional_product_dict.items():
		add_current_product = True
		for local_product in extracted_product_list:
			if current_product_name.lower() in local_product:
				add_current_product = False
		if add_current_product:
			extracted_product_list.append(current_product_name.lower())
			ordered_product_dict[current_product_name] = product_details
		

	return ordered_product_dict
	
	
def get_spans(input_text):
	"""
	 tokenize input string and get start and end index of each token using nltk
	"""

	tokens = nltk.word_tokenize(input_text)
	spans_list = []
	offset = 0
	for token in tokens:
		offset = input_text.find(token, offset)
		value = (token, offset, offset + len(token))
		spans_list.append(value)
		offset += len(token)
	return tokens, spans_list


	
def text_flookup_dict(v_input_text, v_filter_list=None):
	"""
	#Input is a free form text. Exact Matching Lookup on dictionary and fetching other values in the row of the dictionary.
	"""

	regex = re.compile('[%s]' % re.escape(string.punctuation.replace('/', '')))

	# initialize reaction dict
	reaction_dict = dict()

	# to add matched reaction and its start index in text
	reaction_indices_dict = dict()

	# convert input text to lower case
	v_input_text_lower = v_input_text.lower()

	# Tokenizer needed to have words without punctuation marks
	v_input_text_lower_punc = regex.sub(' ', v_input_text_lower)
	v_input_text_tokens, v_input_text_spans = get_spans(
		v_input_text_lower_punc)  # nltk.word_tokenize(v_CIOMS_text_lower)
	# create filter dict
	x_fd = filter_dict(v_filter_list)

	# loop over all reactions, sorted by length of reactions (largest reaction will come first)
	for current_reaction in sorted(x_fd, key=len, reverse=True):
		# lower, remove punctuations and tokenize current reaction
		current_reaction_lower = current_reaction.lower()
		current_reaction_punc = regex.sub(' ', current_reaction)
		reaction_tokens = nltk.word_tokenize(current_reaction_punc)

		f_score = None
		matched_string = ''
		reaction_part_added = False
		# if current reaction has only one token
		# and exact token present in input text then add it
		if len(reaction_tokens) == 1 and current_reaction_lower in v_input_text_tokens:
			pattern_current_reaction_lower_punc = r'\b' + current_reaction_punc.lower() + r'\b'
			matched = re.search(pattern_current_reaction_lower_punc, v_input_text_lower_punc)
			if matched:

				reaction_start_index = matched.start(0)
				# check if current reaction is part of any previously identified reactions
				# using indices of current reaction and previously identified reactions
				for key, start_index in reaction_indices_dict.items():
					if start_index <= reaction_start_index <= start_index + len(key):
						reaction_part_added = True
				if reaction_part_added is False:
					matched_string = matched.group(0)
					f_score = [current_reaction, 100]
		else:
			word_matched_count = 0
			reaction_start_index = sys.maxsize
			reaction_end_index = 0
			for current_token in reaction_tokens:
				# current reaction token in input text
				current_token = current_token.lower()
				if current_token in v_input_text_tokens:
					# get start and end index of current token in input text
					index = v_input_text_tokens.index(current_token)
					start_index = v_input_text_spans[index][1]  # v_input_upper.find(current_token)
					end_index = v_input_text_spans[index][2]  # start_index + len(current_token)
					# update current reaction start and end index in input text
					if start_index < reaction_start_index:
						reaction_start_index = start_index
					if end_index > reaction_end_index:
						reaction_end_index = end_index
					word_matched_count += 1

			if word_matched_count == len(reaction_tokens) and (reaction_end_index - reaction_start_index) < len(
					current_reaction_punc) + 5:
				# check if current reaction is part of any previously identified reactions
				# using indices of current reaction and previously identified reactions
				for key, start_index in reaction_indices_dict.items():
					if start_index <= reaction_start_index <= start_index + len(key):
						reaction_part_added = True

				# if current reaction is not part of any previously identified reaction then add
				if reaction_part_added is False:
					# if current reaction found in input text then set score as 100
					# else match string using fuzzy match
					if current_reaction_lower == (v_input_text_lower[reaction_start_index:reaction_end_index]).strip():
						f_score = [current_reaction, 100]
						matched_string = current_reaction_lower
					else:
						f_score = fuzzy_lookup.extract_best_match(v_input_text[reaction_start_index:reaction_end_index],
																  current_reaction,
																  scorer=fuzzy_lookup.similarity_using_set)
						matched_string = v_input_text[reaction_start_index:reaction_end_index]

		# if matched string is not null then add
		if matched_string.strip() != '':
			serious_reaction = ime_elookup(x_fd[current_reaction][2])
			reaction_dict[current_reaction] = [reaction_start_index, matched_string,
											   f_score[1], current_reaction, x_fd[current_reaction][0],
											   x_fd[current_reaction][1], x_fd[current_reaction][2],
											   x_fd[current_reaction][3], x_fd[current_reaction][4],
											   x_fd[current_reaction][5], x_fd[current_reaction][6],
											   x_fd[current_reaction][7], x_fd[current_reaction][8],
											   x_fd[current_reaction][9], serious_reaction]

			# add current reaction identified and its start index
			reaction_indices_dict[matched_string] = reaction_start_index

	return reaction_dict
	

## Country Lookup
with open('dictionary/dictionary_Country.csv') as f_country:
    next(f_country, None)  # Ignoring the header
    r = csv.reader(f_country, delimiter=',', quotechar='"')
    country_list = list(r)
f_country.close()


def country_elookup(v_input=''):
    """
    check and convert country in standard form
    """
    country_code = ''
    country_name = ''
    try:
        v_input = v_input.strip()
        v_input_lower = v_input.lower()
        for row in country_list:
            # get initials and lower it
            initials = ''
            for x in row[1].lower().split():
                if x not in ['of', 'and']:
                    initials += x[0]
            # initials = ''.join([x[0] for x in row[1].split()])
            initials_lower = initials.lower()
            # check if input has only 2 letters
            if len(v_input) == 2:
                if v_input_lower == row[10].lower() or v_input_lower == initials_lower:
                    country_code = row[10]
                    country_name = row[1]
                    break
            # check if input has only 3 letters
            elif len(v_input) == 3:
                if v_input_lower == row[11].lower() or v_input_lower == initials_lower:
                    country_code = row[10]
                    country_name = row[1]
                    break
            # else
            elif len(v_input) > 3:
                if row[1].lower() in v_input_lower or row[2].lower() == v_input_lower:
                    country_code = row[10]
                    country_name = row[1]
                    break
    except Exception as ex:
        logging.exception("message")
    return country_name
	
def check_if_indication(medical_terms_list, drug):
	print('Med terms list:',medical_terms_list)
	with open('dictionary/dictionary_indication.csv') as f:
		next(f, None)  # Ignoring the header
		r = csv.reader(f, delimiter='|', quotechar='"')
		indications = list(r)
	f.close()
	#print('Indications :',indications)
	indication_list = [row[1] for row in indications if row[0].strip().lower() in drug.strip().lower()]
	
	event_list = list()
	filtered_indications = list()
	for med_term in medical_terms_list:
		for indication in indication_list:
			match_ratio = fuzz.partial_ratio(indication, med_term)
			if match_ratio < 80 and med_term not in event_list:
				event_list.append(med_term)
				break
			else:
				filtered_indications.append(med_term)
				
	return filtered_indications,event_list
	
	
	
	
	
