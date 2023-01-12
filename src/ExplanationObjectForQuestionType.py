import src.analyzer
import pandas as pd
import matplotlib.pyplot as plt

class ExplanationObjectForQuestionType:
    
    def __init__(self, question_type, data_points_df):
        
        # VARIABLES
        
        # question type
        self.qustion_type = question_type
        
        # relevant data
        self.data_df = self._get_relevant_data(data_points_df)
        
        # -- answer lengt
        self.answer_length_info_series = self._get_answer_length_range() 
        
        # -- details about the token importance from differnt perspectives
        self.token_importance_perspectives_dict = self._get_token_details()
        
        # -- details about the end token
        
        # -- details about important word in the answer
        
    def __str__(self):
        
        res = f"""
        === {self.qustion_type.upper()} ===
        Die Antwortlänge beträgt in 75% der Fälle zwischen {self.answer_length_info_series[0]} und {self.answer_length_info_series[1]} Worten.
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
        
        print(f'out of {len(answer_context_df)} data points, {len(answer_context_df) - len(relevant_answer_lengths_df)} have been deleted, because GELECTRA could not finde a sufficent answer. {len(relevant_answer_lengths_df)} data point remaining')
    
        return relevant_answer_lengths_df
    
    def _get_answer_length_range(self):
        

        # remove outliers because GELECTRA sometimes findes no answer or answers with the full context as output
        q_low = self.data_df["prediction_length"].quantile(0.125)
        q_hi  = self.data_df["prediction_length"].quantile(0.875)

        df_filtered = self.data_df[(self.data_df["prediction_length"] < q_hi) & (self.data_df["prediction_length"] > q_low)]

        # return series object containing count, mean, std, min, max, 25%, 50% and 75% quantiles
        details =  df_filtered['prediction_length'].describe()

        return [details['min'], details['max']]

        # in case I wanted to use boxplots and the whiskers values as bounds
        # if mode == 'box_plot':
        #     box_plot_answer_lengths = plt.boxplot(self.data_df['prediction_length'].tolist())
        #
        #     return [int(item.get_ydata()[1]) for item in box_plot_answer_lengths['whiskers']]
    
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