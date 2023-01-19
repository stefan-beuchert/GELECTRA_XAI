import src.analyzer
import config
from src.analyzer import get_frequencie_derivation_as_data_frame

import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ExplanationObjectForQuestionType:
    
    def __init__(self, question_type, data_points_df):
        
        # VARIABLES
        
        # question type
        self.question_type = question_type
        
        # relevant data
        self.data_df = self._get_relevant_data(data_points_df)
        
        # -- details about the token importance from different perspectives
        self.token_importance_perspectives_dict = self._get_token_details()
        
        # -- data for to str method
        self.to_str_data = self._get_important_data_for_to_str_method()
        
        # -- details about important word in the answer
        
    def __str__(self):
        
        res = f"""
        GELECTRA gibt bei ähnlichen Fragen mit 75% Wahrscheinlichkeit eine Antwort zwischen {int(self.to_str_data['min'])} und {int(self.to_str_data['max'])} Worten.
        Dabei stellt das Start-Wort in {int(self.to_str_data['top_pos_for_start_token_probability'] + self.to_str_data['second_top_pos_for_start_token_probability'])} % der Fälle ein {self.to_str_data['top_pos_for_start_token']} oder {self.to_str_data['second_top_pos_for_start_token']} da.
        Für das End-Wort wählt GELECTRA {self.to_str_data['top_pos_for_end_token']} oder {self.to_str_data['second_top_pos_for_end_token']} in {int(self.to_str_data['top_pos_for_end_token_probability'] + self.to_str_data['second_top_pos_for_end_token_probability'])} % der Fälle.
        
        Generell spricht GELECTRA {self.to_str_data['top_pos_tag']} (+{int(self.to_str_data['pos_increase'] * 100)}%) und {self.to_str_data['top_ner_tag']} (+{int(self.to_str_data['ner_increase'] * 100)}%) eine überdurchschnittlich hohe Relevanz bei der Auswahl der Antwort zu.
        """
        
        return res
    
    def visualize(self):
        self.data_df.groupby(['prediction_length']).size().plot(kind = "bar", 
                                                                xlabel = 'token length', 
                                                                ylabel = 'occurences',
                                                                title = 'span length'
                                                               )
        
        for key in list(self.token_importance_perspectives_dict.keys()):
            
            token_sub_set_description = key[0]
            x_value = key[1] # aggragation dimension (token, pos or ner)
            y_value = key[2] # word importance (total frequency or tfidf*lime_score)
            
            # check if df is emtpy, since some combinations don't have any words in them 
            # (end_tokens_negative_df, word, weight) is one of them. By using 1% percentile, it stays emtpy
            if not self.token_importance_perspectives_dict[key].empty:
                self.token_importance_perspectives_dict[key].plot(
                    kind = 'bar',
                    x = x_value, 
                    y = y_value,
                    title = token_sub_set_description
                    )
            else:
                print(f"No data for {token_sub_set_description} {x_value} {y_value}")
                
    def _get_relevant_data(self, data):
        answer_context_df = data
        data['prediction_length'] = answer_context_df.apply(
            lambda row:len(row['prediction'].split()), axis = 1
        )
        answer_context_df['context_length'] = answer_context_df.apply(lambda row: len(row['context'].split()), axis = 1)
        
        relevant_answer_lengths_df = answer_context_df[(answer_context_df['prediction_length'] > 0) 
                                                       & (answer_context_df['prediction_length'] < answer_context_df['context_length'])]
        
        print(f'out of {len(answer_context_df)} data points, {len(answer_context_df) - len(relevant_answer_lengths_df)} have been deleted, because GELECTRA could not find a sufficient answer. {len(relevant_answer_lengths_df)} data point remaining')
    
        return relevant_answer_lengths_df

    def _get_important_data_for_to_str_method(self):

        def get_answer_length_range(data_df):
            # remove outliers because GELECTRA sometimes findes no answer or answers with the full context as output
            q_low = data_df["prediction_length"].quantile(0.125)
            q_hi = data_df["prediction_length"].quantile(0.875)

            df_filtered = data_df[
                (data_df["prediction_length"] < q_hi) & (data_df["prediction_length"] > q_low)]

            # return series object containing count, mean, std, min, max, 25%, 50% and 75% quantiles
            details = df_filtered['prediction_length'].describe()

            return {'min' : details['min'], 'max' : details['max']}

            # in case I wanted to use boxplots and the whiskers values as bounds
            # if mode == 'box_plot':
            #     box_plot_answer_lengths = plt.boxplot(self.data_df['prediction_length'].tolist())
            #
            #     return [int(item.get_ydata()[1]) for item in box_plot_answer_lengths['whiskers']]

        def get_pos_preferences_for_start_and_end_tokens(data_df):

            # get prediction tokens
            prediction_list = data_df['prediction'].tolist()
            prediction_list_split = [p.split() for p in prediction_list]

            prediction_list_split_nan = []
            for p in prediction_list_split:
                if len(p) == 0:
                    prediction_list_split_nan.append([None])
                else:
                    prediction_list_split_nan.append(p)

            # get prediction pos
            nlp = spacy.load("de_core_news_sm")

            pos_prediction_list = []

            for p in prediction_list_split_nan:
                if p[0] is not None:
                    nlp_tags = nlp(' '.join(p))
                    list_pos_tags = [i.pos_ for i in nlp_tags]

                    pos_prediction_list.append(list_pos_tags)

                else:
                    pos_prediction_list.append(p)

            data_df['start_token'] = [p[0] for p in prediction_list_split_nan]
            data_df['end_token'] = [p[-1] for p in prediction_list_split_nan]

            data_df['start_token_pos'] = [p[0] for p in pos_prediction_list]
            data_df['end_token_pos'] = [p[-1] for p in pos_prediction_list]

            data_df['start_token'].fillna(value=np.nan, inplace=True)
            data_df['end_token'].fillna(value=np.nan, inplace=True)
            data_df['start_token_pos'].fillna(value=np.nan, inplace=True)
            data_df['end_token_pos'].fillna(value=np.nan, inplace=True)

            start_token_pos_df = data_df['start_token_pos'].value_counts().to_frame()
            start_token_pos_df['percent'] = (start_token_pos_df['start_token_pos'] / start_token_pos_df[
                'start_token_pos'].sum()) * 100

            end_token_pos_df = data_df['end_token_pos'].value_counts().to_frame()
            end_token_pos_df['percent'] = (end_token_pos_df['end_token_pos'] / end_token_pos_df[
                'end_token_pos'].sum()) * 100

            start_token_pos_df = start_token_pos_df.sort_values(by=['percent'], ascending=False)
            end_token_pos_df = end_token_pos_df.sort_values(by=['percent'], ascending=False)

            res = {
                'top_pos_for_start_token': start_token_pos_df.index[0],
                'top_pos_for_start_token_probability': start_token_pos_df.iloc[0]['percent'],

                'second_top_pos_for_start_token': start_token_pos_df.index[1],
                'second_top_pos_for_start_token_probability': start_token_pos_df.iloc[1]['percent'],

                'top_pos_for_end_token': end_token_pos_df.index[0],
                'top_pos_for_end_token_probability': end_token_pos_df.iloc[0]['percent'],

                'second_top_pos_for_end_token': end_token_pos_df.index[1],
                'second_top_pos_for_end_token_probability': end_token_pos_df.iloc[1]['percent'],
            }

            return res

        answer_length_dict = get_answer_length_range(self.data_df)
        start_end_token_pos_dict = get_pos_preferences_for_start_and_end_tokens(self.data_df)

        pos_frequencies = get_frequencie_derivation_as_data_frame('pos_tag', self.data_df, config.POS_tag_list, 'self.question_type')
        pos_frequencies = pos_frequencies.reset_index(drop=True)
        ner_frequencies = get_frequencie_derivation_as_data_frame('ner_tag', self.data_df, config.NER_tag_list, 'self.question_type')
        ner_frequencies = ner_frequencies.reset_index(drop=True)

        frequencies_insights_dict = {
            # ner
            'top_ner_tag' :  ner_frequencies.loc[0]['tag'],
            'ner_increase': ner_frequencies.loc[0]['percentage difference'],

            # pos
            'top_pos_tag': pos_frequencies.loc[0]['tag'],
            'pos_increase': pos_frequencies.loc[0]['percentage difference']
        }

        all_to_str_infos_dict = {**answer_length_dict, **frequencies_insights_dict, **start_end_token_pos_dict}

        return all_to_str_infos_dict
    
    def _get_token_details(self):
        
        question_tokens_with_weights_dict = src.analyzer.sort_tokens_in_categories(self.data_df['explanation'].tolist())
        percentile_1 = int(len(self.data_df) * 0.01)
        
        # res dict with all possible combinations
        # the possible combinations:
        # 6: start_tokens, end_tokens and both groups with only pos and neg tokens
        # 3: token, pos or ner level
        # 2: weighted average or total frequency
        # makes a total of 6*3*2 = 36 perspectives
        res_dict = {}
        
        # for each token_subset (start, end, pos and neg)
        for token_sub_set in list(question_tokens_with_weights_dict.keys()):
            # for each abstraction layer
            for abstraction_layer in ['word', 'pos_tag', 'ner_tag']:
                # for the different token weights (by tfidf*lime or occurance)
                for token_weight_representation in ['weight', 'count']:
                    
                    # calculate the sorted view on the data
                    data_perspective_sorted = src.analyzer.get_frequencie(question_tokens_with_weights_dict[token_sub_set],
                                                                         max_tokens = 15, 
                                                                         mode = token_weight_representation, 
                                                                         target = abstraction_layer, 
                                                                         remove_stop_words = True, 
                                                                         min_count = percentile_1)
                    # save perspective in res dict with 3 keys
                    res_dict[(token_sub_set, 
                              abstraction_layer, 
                              token_weight_representation)] = data_perspective_sorted
        
        return res_dict