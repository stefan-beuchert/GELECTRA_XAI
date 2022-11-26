import config

import re


def get_full_explanation_for_data_point(lime_explainer, nlp, question, context):
    # get lime explanation
    lime_input = question + " " + context
    lime_explanation = lime_explainer.explain(lime_input)

    # tokens
    lime_tokens = re.split(config.LIME_REGEX_FOR_TOKENIZATION, lime_input)

    # positions
    postion_ids = list(range(len(lime_tokens)))

    # question_or_context token
    question_tokens = re.split(config.LIME_REGEX_FOR_TOKENIZATION, question)
    queston_length = len(question_tokens) - 1  # substract 1 for the additonal empty token at the ende
    question_or_context_tags = ['question'] * queston_length + ['context'] * (len(postion_ids) - queston_length)

    # pos, ner and iob
    nlp_tags = nlp(' '.join(lime_tokens))

    # pos
    list_pos_tags = [i.pos_ for i in nlp_tags]

    # ner
    ner_tags = [token.ent_type_ for token in nlp_tags]

    # iob
    iob_tags = [token.ent_iob_ for token in nlp_tags]

    # TODO: For some reasons some token lists don't match up with the list of pos, ner and iob
    # The length of the list differs by one index. A possible explanation is the additional '' (empty token)
    # that is found at the end of every token_list. A solution is to add an addtional token at the and
    # of NER, POS and IOB, which reloves the issue. But in about 4 out of 100 data points there is the
    # extra token at the end of lime_tokens BUT it still adds up with the length of the other lists.
    # So always adding a '' to NER POS and IOB is not the solution. An investigation should be done here
    # to identify the exact reason, but for new a quick fix is to just add '' at the end of the lists, if the
    # lengths don't match up. This seems to resolve the use, but bares the risk of creating NER, POS and IOB tags
    # that are off by one to the correct token
    if len(lime_tokens) > len(list_pos_tags):
        list_pos_tags.append('')  # doing this to match the token length
        ner_tags.append('')  # doing this to match the token length
        iob_tags.append('O')  # doing this to match the token length

    # start token weights
    start_token_weigths = lime_explanation.as_map()[0]

    start_token_weights_map = [None] * len(lime_tokens)
    for lime_weight in start_token_weigths:
        start_token_weights_map[lime_weight[0]] = lime_weight[1]

    # end token weights
    end_token_weigths = lime_explanation.as_map()[1]

    end_token_weights_map = [None] * len(lime_tokens)
    for lime_weight in end_token_weigths:
        end_token_weights_map[lime_weight[0]] = lime_weight[1]

    # tmp
    #     print(len(lime_tokens))
    #     print(len(postion_ids))
    #     print(len(question_or_context_tags))
    #     print(len(list_pos_tags))
    #     print(len(ner_tags))
    #     print(len(iob_tags))
    #     print(len(start_token_weights_map))
    #     print(len(end_token_weights_map))

    #     print(lime_tokens)
    #     print(ner_tags)

    # check if all components have the same length. if not raise an error
    if not (len(lime_tokens)
            == len(postion_ids)
            == len(question_or_context_tags)
            == len(list_pos_tags) == len(ner_tags)
            == len(iob_tags)
            == len(start_token_weights_map)
            == len(end_token_weights_map)):
        raise ValueError('The length of at least one property did not match up with the token length. '
                         'Impossible to map properties to tokens')

    # create a final data object with all the information

    final_data_object = []

    for postion_id in postion_ids:
        token_description = {
            'position_id': postion_id,
            'lime_token': lime_tokens[postion_id],
            'question_or_context_tag': question_or_context_tags[postion_id],
            'start_token_weight': start_token_weights_map[postion_id],
            'end_token_weight': end_token_weights_map[postion_id],
            'pos_tag': list_pos_tags[postion_id],
            'ner_tag': ner_tags[postion_id],
            'iob_tag': iob_tags[postion_id]
        }

        final_data_object.append(token_description)

    return final_data_object