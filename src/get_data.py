import json
import pandas as pd
import warnings
import unicodedata

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

def enhance_data_with_question_type(data_df):
    # split into question types
    def identify_question_type(question):
        question = question.lower()
        if question[-1] == '?': question = question[:-1]  # remove '?' from question (if there is one)

        question_type = None

        # check for 'wie viel' as the only w-question that contains two 2 tokens
        if 'wie viel' in question:
            question_type = 'wie viel'

        # if 'wie viel' is identified, no need to check for other question words
        else:
            # split question into tokens
            question_tokens = question.split(' ')

            # turns out that all questions have the question word at the beginning of the question.
            # therefore we only have to look at the first token
            # question_tokens = [question_tokens[0]]

            question_words_list = [
                # https://de.wikipedia.org/wiki/W-Wort
                'wer', 'welche', 'welcher',
                'wem', 'wen', 'welchen', 'welchem',
                'wessen',
                'was', 'welches',
                'warum', 'weshalb', 'weswegen', 'wieso',
                'wie', 'wieweit',  # 'wie viel'
                'wofür', 'wozu', 'womit', 'wodurch', 'worum', 'worüber', 'wobei', 'wovon', 'woraus',
                'wo', 'wogegen',
                'wohin', 'woher',
                'woran', 'worin', 'worauf', 'worunter', 'wovor', 'wohinter', 'woneben',
                'wann',
                # below words that have been added by looking at the questions that are not covered by the word above
                'wonach', 'inwiefern'
            ]

            # check for other w-quetions
            for question_word in question_words_list:
                if question_word in question_tokens:
                    if (question_type is not None):
                        # print(f'{question} - has multiple question types')
                        question_type = 'conflict'
                    else:
                        question_type = question_word

        if question_type is None:
            # print(f'{question} - has no question type')
            question_type = 'undefined'

        return question_type

    question_list = data_df['question'].tolist()
    question_type_list = [identify_question_type(q) for q in question_list]
    data_df['question_type'] = question_type_list

    return data_df

def enhance_data_with_correctness_of_the_prediction(data_df):
    def check_if_prediction_is_correct(row):

        prediction = str.lower(row['prediction'])
        possible_answers_list_of_dicts = row['answers']

        possible_answers_list = [str.lower(answer_dict['text']) for answer_dict in possible_answers_list_of_dicts]

        for possible_answer in possible_answers_list:

            # decode data to reduce errors like "−27\xa0°C" which should be "−27 °C"
            possible_answer_decoded = unicodedata.normalize("NFKD", possible_answer)

            # also decode the prediction, just to make sure both have the same format
            prediction_decoded = unicodedata.normalize("NFKD", prediction)

            if possible_answer_decoded == prediction_decoded:
                return True

        return False

    data_df['prediction_correct'] = data_df.apply(lambda row: check_if_prediction_is_correct(row), axis=1)

    return data_df