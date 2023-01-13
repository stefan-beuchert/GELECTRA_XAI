from IPython.display import display_html 

import pandas as pd
import nltk
import spacy

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
german_stop_words = nltk.corpus.stopwords.words('german')

spacy_nlp = spacy.load("de_core_news_sm")

def sort_tokens_in_categories(explanations_list):
    # This function will take a list of explanations and return tokens with weights in 6 categories:
    # - all start tokens
    # - all end tokens
    # - all positive start tokens
    # - all negative start tokens
    # - all positive end tokens
    # - all negative end tokens
    
    # INPUT: List of explanation dataframes
    # OUTPUT: The different categories are saved as df and stored in a dict which will be returned
    
    # dict to save differnt dfs
    tokens_with_weights_dict = {}

    # create dfs with 'words' and 'weights' (words don't have to be unique)
    ## start token ##
    start_tokens_df_test = pd.concat(
        [exp[
            ~exp['start_lime_tfidf_score'].isnull() # condition to select all not NaN values
        ] for exp in explanations_list]
    )

    # select and rename specific columns
    #start_tokens_df_test = start_tokens_df_test[['token_lower', 'start_lime_tfidf_score']] # subset of columns that should be in the output
    start_tokens_df_test = start_tokens_df_test.rename({'token_lower': 'word', 'start_lime_tfidf_score': 'weight'}, axis='columns')

    ## end token ##
    end_tokens_df_test = pd.concat(
        [exp[
            ~exp['end_lime_tfidf_score'].isnull() # condition to select all not NaN values
        ] for exp in explanations_list]
    )

    # select and rename specific columns
    #end_tokens_df_test = end_tokens_df_test[['token_lower', 'end_lime_tfidf_score']] # subset of columns that should be in the output
    end_tokens_df_test = end_tokens_df_test.rename({'token_lower': 'word', 'end_lime_tfidf_score': 'weight'}, axis='columns')

    # save
    tokens_with_weights_dict['start_tokens_df'] = start_tokens_df_test
    tokens_with_weights_dict['end_tokens_df'] = end_tokens_df_test

    # add sub sets with only pos and only neg values
    tokens_with_weights_dict['start_tokens_positive_df'] = start_tokens_df_test[start_tokens_df_test['weight'] > 0]
    tokens_with_weights_dict['start_tokens_negative_df'] = start_tokens_df_test[start_tokens_df_test['weight'] < 0]

    tokens_with_weights_dict['end_tokens_positive_df'] = end_tokens_df_test[end_tokens_df_test['weight'] > 0]
    tokens_with_weights_dict['end_tokens_negative_df'] = end_tokens_df_test[end_tokens_df_test['weight'] < 0]
    
    return tokens_with_weights_dict


def get_frequencie(lime_explanation_df, max_tokens, mode, target, remove_stop_words, min_count):
    
    # input should be a df with the columns 'word' and 'weight'
    
    def clean_data(data_df):
        if remove_stop_words:
            data_df = data_df[~data_df['word'].isin(german_stop_words)]
            
        return data_df
        
    # prepare data
    lime_explanation_df_clean = clean_data(lime_explanation_df)
    
    # get frequencies
    frequencies_df = lime_explanation_df_clean.groupby(target)['weight'].agg(['sum','count'])
    frequencies_df = frequencies_df[frequencies_df['count'] >= min_count]
    frequencies_df = frequencies_df.reset_index()
    
    # return total of most frequent words, weights not included
    if mode == 'count':
        return frequencies_df[[target, 'count']].sort_values(by=['count'], ascending=False).head(max_tokens)
        
    # return most important word by calculation the l2 (or euclidean) distance for a vectore (count, sum) to the origin (0,0)
    elif mode == 'weight':
        frequencies_df['weight'] = abs(frequencies_df['sum'] / frequencies_df['count'])
        #frequencies_df['weight'] = frequencies_df.apply(lambda row: np.linalg.norm(np.array((row['count'], row['sum']))), axis=1)
        return frequencies_df[[target, 'weight', 'count']].sort_values(by=['weight'], ascending=False).head(max_tokens)
        
    # res is a touple with (word_list, freq_list)
    else:
        print("WARNING, wrong mode")
        
def display_frequencies(tokens_dict, max_tokens = 15, mode = 'count', target = 'word', remove_stop_words = False, min_count = 0):
    start_token_word_freq_df = get_frequencie(tokens_dict['start_tokens_df'], max_tokens, mode, target, remove_stop_words, min_count)
    start_token_positive_word_freq_df = get_frequencie(tokens_dict['start_tokens_positive_df'], max_tokens, mode, target, remove_stop_words, min_count)
    start_token_negative_word_freq_df = get_frequencie(tokens_dict['start_tokens_negative_df'], max_tokens, mode, target, remove_stop_words, min_count)

    start_token_styler = start_token_word_freq_df.style.set_table_attributes("style='display:inline'").set_caption('start total')
    start_token_positive_styler = start_token_positive_word_freq_df.style.set_table_attributes("style='display:inline'").set_caption('start positive')
    start_token_negative_styler = start_token_negative_word_freq_df.style.set_table_attributes("style='display:inline'").set_caption('start negative')

    end_token_word_freq_df = get_frequencie(tokens_dict['end_tokens_df'], max_tokens, mode, target, remove_stop_words, min_count)
    end_token_positive_word_freq_df = get_frequencie(tokens_dict['end_tokens_positive_df'], max_tokens, mode, target, remove_stop_words, min_count)
    end_token_negative_word_freq_df = get_frequencie(tokens_dict['end_tokens_negative_df'], max_tokens, mode, target, remove_stop_words, min_count)

    end_token_styler = end_token_word_freq_df.style.set_table_attributes("style='display:inline'").set_caption('end total')
    end_token_positive_styler = end_token_positive_word_freq_df.style.set_table_attributes("style='display:inline'").set_caption('end positive')
    end_token_negative_styler = end_token_negative_word_freq_df.style.set_table_attributes("style='display:inline'").set_caption('end negative')

    display_html(start_token_styler._repr_html_() + start_token_positive_styler._repr_html_() + start_token_negative_styler._repr_html_() + 
                 end_token_styler._repr_html_() + end_token_positive_styler._repr_html_() + end_token_negative_styler._repr_html_(), raw=True)

def get_frequencies_as_data_frame_in_comparison_to_expected_frequencies(token_dimension, data_df, tag_list):

    # token_dimension: string value. Either 'token', 'pos_tag' or 'ner_tag' depending of the desired abstraction layer
    # data_df: data as data frame from 'data/Data_Preparation folder
    # tag_list: list of strings containing all possible tags

    # get list of all explanations (containing every data point with info about each token)
    explanation_list = data_df.explanation.tolist()

    # create empty dicts to count POS tags
    general_counter_for_tag = {tag: 0 for tag in tag_list}
    lime_highlights_for_start_tokens__counter_for_tag = {tag: 0 for tag in tag_list}
    lime_highlights_for_end_tokens__counter_for_tag = {tag: 0 for tag in tag_list}

    # for each token in each data point
    for explanation in explanation_list:

        # get lime highlights for start and end token subsets
        lime_highlights_for_start_token = explanation[explanation['start_token_weight'].notnull()]
        lime_highlights_for_end_token = explanation[explanation['end_token_weight'].notnull()]

        # get total count of each POS tag for that data point
        total_count_of_each_tag = explanation[token_dimension].value_counts()
        total_count_of_each_tag_for_start_token_highlights = lime_highlights_for_start_token[token_dimension].value_counts()
        total_count_of_each_tag_for_end_token_highlights = lime_highlights_for_end_token[token_dimension].value_counts()

        # convert panda.Series to dicts for better extraction of values
        total_count_of_each_tag_dict = total_count_of_each_tag.to_dict()
        total_count_of_each_tag_for_start_token_highlights_dict = total_count_of_each_tag_for_start_token_highlights.to_dict()
        total_count_of_each_tag_for_end_token_highlights_dict = total_count_of_each_tag_for_end_token_highlights.to_dict()

        # save POS counts in designated dicts
        for tag in total_count_of_each_tag_dict:
            general_counter_for_tag[tag] += total_count_of_each_tag_dict[tag]

        for tag in total_count_of_each_tag_for_start_token_highlights_dict:
            lime_highlights_for_start_tokens__counter_for_tag[tag] += \
            total_count_of_each_tag_for_start_token_highlights_dict[tag]

        for tag in total_count_of_each_tag_for_end_token_highlights_dict:
            lime_highlights_for_end_tokens__counter_for_tag[tag] += \
            total_count_of_each_tag_for_end_token_highlights_dict[tag]

    # safe data in df
    frequencies_df = pd.DataFrame([general_counter_for_tag,
                                   lime_highlights_for_start_tokens__counter_for_tag,
                                   lime_highlights_for_end_tokens__counter_for_tag])

    frequencies_df = frequencies_df.T
    frequencies_df = frequencies_df.rename({0: 'general total', 1: 'start token total', 2: 'end token total'},
                                           axis='columns')

    frequencies_df['general percentage'] = frequencies_df['general total'] / frequencies_df['general total'].sum()
    frequencies_df['start token percentage'] = frequencies_df['start token total'] / frequencies_df[
        'start token total'].sum()
    frequencies_df['end token percentage'] = frequencies_df['end token total'] / frequencies_df['end token total'].sum()

    frequencies_df['start token deviation from general'] = (frequencies_df['start token percentage'] - frequencies_df[
        'general percentage']) / frequencies_df['general percentage']
    frequencies_df['end token deviation from general'] = (frequencies_df['end token percentage'] - frequencies_df[
        'general percentage']) / frequencies_df['general percentage']

    frequencies_df['start token total deviation'] = frequencies_df['start token total'] - (
                frequencies_df['general percentage'] * frequencies_df['start token total'].sum())
    frequencies_df['end token total deviation'] = frequencies_df['end token total'] - (
                frequencies_df['general percentage'] * frequencies_df['end token total'].sum())

    return frequencies_df