import json
import pandas as pd
import warnings

# config
local_path_folder = './data/GermanQuAD_raw/'
path_train_data = local_path_folder + 'GermanQuAD_train.json'
path_test_data = local_path_folder + 'GermanQuAD_test.json'

# helper functions
##################

def GermanQuAD_format_to_df(json_data):
    question_answers_pairs_list = []
    for i in json_data['data']:
        for j in i['paragraphs'][0]['qas']:
            question_answers_pairs_list.append(
                {
                    'question_id' : j['id'],
                    'question' : j['question'],
                    'answers' : j['answers'],
                    'context' : i['paragraphs'][0]['context'],
                    'document_id' : i['paragraphs'][0]['document_id'],
                    'is_impossible' : j['is_impossible']
                }
            )
    question_answers_pairs_df = pd.DataFrame(question_answers_pairs_list)
            
    return question_answers_pairs_df

# main functions
################

def get_data():
    
    # load data from source
    GermanQuAD_train_json = json.loads(open(path_train_data, encoding="utf8").read())
    GermanQuAD_test_json = json.loads(open(path_test_data, encoding="utf8").read())
    
    # convert to df
    GermanQuAD_train_df = GermanQuAD_format_to_df(GermanQuAD_train_json)
    GermanQuAD_test_df = GermanQuAD_format_to_df(GermanQuAD_test_json)
    
    # add usage label
    GermanQuAD_train_df['usage'] = 'train'
    GermanQuAD_test_df['usage'] = 'test'
    
    # combine to a single df
    res = pd.concat([GermanQuAD_train_df, GermanQuAD_test_df], axis=0)
    
    # shuffle the rows
    res.sample(frac=1)
    
    # check for duplicates
    if res['question_id'].duplicated().any():
        warnings.warn("WARNING, there are duplicates in the dataset")
    
    return res