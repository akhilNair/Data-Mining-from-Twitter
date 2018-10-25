import re
import editdistance
from difflib import SequenceMatcher
from nltk.stem import PorterStemmer, WordNetLemmatizer

# constants
#weighted_scorer = 'weighted_scorer'
similarity_using_sort = 'word_sort_scorer'
similarity_using_set = 'word_set_scorer'


def _standardize_string(input_string, stemming=False, lemmatization=False):
    """
     Do following operations on string
     1. replace all non letter and non number characters with white space in string
     2. convert into lower case
     3. trim heading and trailing white spaces
    """
    if input_string is None:
        return ""
    else:
        # string conversion, lowercase, strip, replace non-alphanumeric by space
        output_string = re.compile(r'[^a-zA-Z0-9]').sub(' ', str(input_string).strip().lower()) 

    # apply stemming on each word
    if stemming:
        port = PorterStemmer()
        output_string = ' '.join(port.stem(word) for word in output_string.split())

    # apply lemmatization on each word
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        output_string = ' '.join(lemmatizer.lemmatize(word) for word in output_string.split())

    return output_string


##def _levenshtein_edit_distance(str1, str2):
##    """   levenshtein implementation from link 'https://rosettacode.org/wiki/Levenshtein_distance#Python'  """
##    if str1 == str2:
##        return 0
##    elif len(str1) == 0:
##        return len(str2)
##    elif len(str2) == 0:
##        return len(str1)
##
##    m = len(str1)
##    n = len(str2)
##    lensum = float(m + n)
##    d = []
##    for i in range(m+1):
##        d.append([i])
##    del d[0][0]
##    for j in range(n+1):
##        d[0].append(j)
##    for j in range(1, n+1):
##        for i in range(1, m+1):
##            if str1[i-1] == str2[j-1]:
##                d[i].insert(j, d[i-1][j-1])
##            else:
##                minimum = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+2)
##                d[i].insert(j, minimum)
##    ldist = d[-1][-1]
##    return ldist


def _calculate_similarity_score(source_string, target_string):
    """ calculate similarity between given string  """
    source_length = len(source_string)
    target_length = len(target_string)
    total_length = source_length + target_length
    if total_length == 0:
        return 100
    edit_distance = editdistance.eval(source_string, target_string) # Levenshtein Distance calculation
    if edit_distance < 0:
        return 0
    score = int(round( 100 * float(total_length - edit_distance) / total_length )) # rounding number to integer
    return score


def _calculate_partial_similarity(string_1, string_2):
    """
     calculate partial ratio between given string
     1. Identify which is small and which is large string
     2. Get matching blocks from large string to small string
     3. If one or more matching blocks found then
     4. For each block get sub string from large string
     5. calculate ratio between small string and sub string from large string
     6. if this ratio if greater than previous block ratio then make it as max ratio
    """

    max_similarity = 0
    #string_1, string_2 = _check_and_convert_input_into_string(string_1, string_2)

    # identify which is small and which is large string
    if len(string_1) > len(string_2):
        small_string = string_2
        large_string = string_1
    else:
        small_string = string_1
        large_string = string_2

    # get matching blocks from large string to small string
    s_matcher = SequenceMatcher(None, small_string, large_string)
    matched_blocks = s_matcher.get_matching_blocks()

    # if one or more matching blocks found then
    for current_block in matched_blocks:
        # if not last block
        if current_block[2] != 0:
            # get sub string from large string
            large_start_index = current_block[1] - current_block[0]
            if large_start_index < 0:
                large_start_index = 0
            large_end_index = len(small_string)
            large_sub_string = large_string[large_start_index : large_end_index]

            # calculate ratio between small string and sub string from large string
            current_similarity = _levenshtein_edit_distance(small_string, large_sub_string)
##            if current_similarity < 0:
##                return None
    
            # if this ratio if greater than previous block ratio then make it as max ratio
            if current_similarity > max_similarity:
                max_similarity = current_similarity

    # return round of value
    max_similarity = int(round(max_similarity * 100)) #rounding to integer
    return max_similarity


def _calculate_similarity_with_word_sort(string_1, string_2, partial=False):
    """  convert strings into words, sort words and  combine and calculate ratio  """
    similarity = 0

    # split string on space
    string_1_words = string_1.split()
    string_2_words = string_2.split()

    # sort splitted strings
    sorted_string_1 = ' '.join(sorted(string_1_words))
    sorted_string_2 = ' '.join(sorted(string_2_words))

    # calculate full or partial ratio
    if partial:
        similarity = _calculate_partial_similarity(sorted_string_1, sorted_string_2)
    else:
        similarity = _calculate_similarity_score(sorted_string_1, sorted_string_2)

    return similarity


def _calculate_similarity_with_word_set(string_1, string_2, partial=False):
    """ convert strings into words, convert list words into set of words and calculate similarity between sets  """

    # split string on space and convert into words
    string_1_words = string_1.split()
    string_2_words = string_2.split()

    # convert list of words into set
    string_1_set_words = set(string_1_words)
    string_2_set_words = set(string_2_words)

    # take intersection of two sets of words
    words_intersection = string_1_set_words.intersection(string_2_set_words)

    # get words which are in string 1 but not in string 2
    words_1_to_2_diff = string_1_set_words.difference(string_2_set_words)

    # get words which are in string 2 but not in string 1
    words_2_to_1_diff = string_2_set_words.difference(string_1_set_words)

    # join all sets of words and create strings
    string_intersection = (' '.join(sorted(words_intersection))).strip()
    string_1_to_2_diff = ' '.join(sorted(words_1_to_2_diff))
    string_2_to_1_diff = ' '.join(sorted(words_2_to_1_diff))

    # combine string intersection with both differences
    combined_1_to_2 = (string_intersection + ' ' + string_1_to_2_diff).strip()
    combined_2_to_1 = (string_intersection + ' ' + string_2_to_1_diff).strip()

    # calculate full or partial ratio
    if partial:
        string_difference_similarity = _calculate_partial_similarity(combined_1_to_2, combined_2_to_1)
##        string_1_to_2_diff_similarity = _calculate_partial_similarity(combined_1_to_2, string_intersection)
##        string_2_to_1_diff_similarity = _calculate_partial_similarity(combined_2_to_1, string_intersection)
    else:
        string_difference_similarity = _calculate_similarity_score(combined_1_to_2, combined_2_to_1)
##        string_1_to_2_diff_similarity = _calculate_similarity_score(string_intersection, combined_1_to_2)
##        string_2_to_1_diff_similarity = _calculate_similarity_score(string_intersection, combined_2_to_1)

##    # get max out of three ratios and return it
##    similarity = max([string_difference_similarity, string_1_to_2_diff_similarity, string_2_to_1_diff_similarity])
    return string_difference_similarity


def extract_values(input_string, content_to_match, scorer=similarity_using_set, cut_off_score=0, stemming=False, lemmatization=False, partial_ratio=False):
    """    return all matches having score greater than cutoff score  """
    output_list = []

    modified_input_string = _standardize_string(input_string,stemming=stemming, lemmatization=lemmatization)
    # if len of content_to_match > 0 and len input_string > 0 then run
    if len(content_to_match) > 0 and len(modified_input_string) > 0:
        # if contents_to_match are dictionary then
        # return a list of tuple (value, ratio, key)
        # if contents_to_match are list then
        # return a list of tuple (value, ratio)
        if isinstance(content_to_match, dict):
            for current_value,matching_string in content_to_match.items():
                # standardize value
                modified_matching_string = _standardize_string(matching_string, stemming=stemming, lemmatization=lemmatization)
                # calculate score based on given scorer
                # if scorer == weighted_scorer:
                # score = _calculate_weighted_ratio(input_string, matching_string, partial_ratio)
                if scorer == similarity_using_sort:
                    score = _calculate_similarity_with_word_sort(modified_input_string, modified_matching_string, partial=partial_ratio)
                elif scorer == similarity_using_set:
                    score = _calculate_similarity_with_word_set(modified_input_string, modified_matching_string, partial=partial_ratio)
                # score is > cut off score then add in output list
                if score >= cut_off_score:
                    output_list.append((matching_string, score, current_value))
                    #yield (matching_string, score, current_value)
        else:
            for index, matching_string in enumerate(content_to_match):
                modified_matching_string = _standardize_string(matching_string, stemming=stemming, lemmatization=lemmatization)
                # calculate score based on given scorer
                # if scorer == weighted_scorer:
                # score = _calculate_weighted_ratio(input_string, matching_string, partial_ratio)
                if scorer == similarity_using_sort:
                    score = _calculate_similarity_with_word_sort(modified_input_string, modified_matching_string, partial=partial_ratio)
                elif scorer == similarity_using_set:
                    score = _calculate_similarity_with_word_set(modified_input_string, modified_matching_string, partial=partial_ratio)
                # score is > cut off score then add in output list
                if score >= cut_off_score:
                    output_list.append((matching_string, score))
                    #yield (matching_string, score)

    return output_list


def extract_top_matches(input_string, content_to_match, scorer=similarity_using_set, top_values_count=5, stemming=False, lemmatization=False, partial_ratio=False):
    """   return top N matches   """
    output = []
    try:
        scores = extract_values(input_string, content_to_match, scorer=scorer, stemming=stemming, lemmatization=lemmatization, partial_ratio=partial_ratio)
        # if matched score are found
        if len(scores) > 0:
            # sort matches in decreasing order
            results = sorted(scores, key=lambda x:x[1], reverse=True)
            # if len of matches < top values count given then return results
            if len(results) < top_values_count:
                output = results
            # else return top matched
            else:
                output = results[:top_values_count]
    except Exception as ex:
        print(ex)

    return output


def extract_best_match(input_string, content_to_match, cutoff_score=0, scorer=similarity_using_set, stemming=False, lemmatization=False, partial_ratio=False):
    """  return best match  """
    output = None
    try:
        scores = extract_values(input_string, content_to_match, scorer=scorer, stemming=stemming, lemmatization=lemmatization, partial_ratio=partial_ratio)
        output = max(scores, key=lambda i: i[1])
        if output[1] < cutoff_score:
            output = None
    except Exception as ex:
        print(ex)
        output = None
    return output

