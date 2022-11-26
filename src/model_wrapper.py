from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch.nn as nn
import torch


# I added this function later (Stefan)
def softmax(logits):
    softmax = nn.Softmax(dim=1)
    return softmax(logits)


class Model:

    def __init__(self, model_name):
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer =  AutoTokenizer.from_pretrained(model_name)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()
        self.model.zero_grad()

        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.ref_token_id = self.tokenizer.pad_token_id
        
    def predict(self, question, context):
    
        encoding = self.tokenizer.encode_plus(question,
                                              context, 
                                              max_length = self.tokenizer.model_max_length,
                                              truncation=True)
        
        # input_ids
        input_ids = torch.tensor([encoding.input_ids], device=self.device)
        # token_type_ids
        token_type_ids = torch.tensor([encoding.token_type_ids], device=self.device)
        # position_ids
        position_ids = torch.tensor([range(len(encoding.input_ids))], device=self.device)
        # attention_mask
        attention_mask = torch.tensor([encoding.attention_mask], device=self.device)


        output = self.model(input_ids, 
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            attention_mask=attention_mask) 

        indices = input_ids[0].detach().tolist()
        all_tokens = self.tokenizer.convert_ids_to_tokens(indices)

        return output, all_tokens, encoding
    
    # Return: List of 3 tupels (Start-Token, Start-Token probability), (End-Token, End-Token probability), (full answer, overall probability)
    def get_predicted_tokens(self, logits, tokens):

        prediction = []

        start_logits = logits.start_logits
        end_logits = logits.end_logits
        all_tokens = tokens
        
        predicted_start_token = all_tokens[torch.argmax(start_logits)]
        predicted_end_token = all_tokens[torch.argmax(end_logits)]
        predicted_answer = str(all_tokens[torch.argmax(start_logits) : torch.argmax(end_logits)+1])

        norm_start_logits = softmax(start_logits)
        norm_end_logits = softmax(end_logits)

        probability_start_token = norm_start_logits[0][torch.argmax(start_logits)].item()
        probability_end_token = norm_end_logits[0][torch.argmax(end_logits)].item()
        probability_answer = torch.mul(probability_start_token,probability_end_token).item()

        prediction.append((predicted_start_token, probability_start_token))
        prediction.append((predicted_end_token, probability_end_token))
        prediction.append((predicted_answer, probability_answer)) 

        return prediction

    def get_answer_string(self,  logits, tokens):
        start_logits = logits.start_logits
        end_logits = logits.end_logits
        all_tokens = tokens
        
        predicted_answer_tokens = all_tokens[torch.argmax(start_logits) : torch.argmax(end_logits)+1]
        answer_string = self.tokenizer.convert_tokens_to_string(predicted_answer_tokens)

        return answer_string
    
######################################################################################
    
    def predict_alt(self, question, context):
        
        question_ids = self.tokenizer.encode(question, add_special_tokens=False)
        text_ids = self.tokenizer.encode(context, add_special_tokens=False)
        
        # check if sequence is to big for the model
        sequence_length = len(question_ids) + len(text_ids)
        
        print(sequence_length)
        if sequence_length >= 511:
            print("break")
            return None, None 
        
        input_ids, seperator_id = self.construct_input_ids(question_ids,text_ids)
        token_type_ids = self.construct_input_token_type_ids(input_ids, seperator_id)
        position_ids = self.construct_input_pos_ids(input_ids)
        print(token_type_ids)
        attention_mask = self.construct_attention_mask(input_ids)

        output = self.model(input_ids, token_type_ids=token_type_ids,position_ids=position_ids, attention_mask=attention_mask)

        indices = input_ids[0].detach().tolist()
        all_tokens = self.tokenizer.convert_ids_to_tokens(indices)

        return output, all_tokens

    def forward(self, inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):  # position = 0 : Start-Token, position = 1 : End-Token
        output = self.model(inputs, token_type_ids=token_type_ids,position_ids=position_ids, attention_mask=attention_mask)
        output = output[position]
        return output.max(1).values


    def construct_input_ids(self, question_ids, text_ids):

        # construct input token ids
        input_ids = [self.cls_token_id] + question_ids + [self.sep_token_id] + text_ids + [self.sep_token_id]
        separator_id = len(question_ids)

        return torch.tensor([input_ids], device=self.device), separator_id

    def construct_input_ids_reference_pair(self, question, context):
        question_ids = self.tokenizer.encode(question, add_special_tokens=False)
        text_ids = self.tokenizer.encode(context, add_special_tokens=False)

        # construct input token ids
        input_ids = [self.cls_token_id] + question_ids + [self.sep_token_id] + text_ids + [self.sep_token_id]

        # construct reference token ids 
        ref_input_ids = [self.cls_token_id] + [self.ref_token_id] * len(question_ids) + [self.sep_token_id] + [self.ref_token_id] * len(text_ids) + [self.sep_token_id]

        return torch.tensor([input_ids], device=self.device), torch.tensor([ref_input_ids], device=self.device), len(question_ids)


    def construct_input_token_type_ids(self, input_ids, seperator_ind=0):
        seq_len = input_ids.size(1)
        token_type_ids = torch.tensor([[0 if i <= seperator_ind else 1 for i in range(seq_len)]], device=self.device)
        return token_type_ids

    def construct_input_token_type_ids_reference_pair(self, input_ids, sep_ind=0):
        seq_len = input_ids.size(1)
        token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=self.device)
        ref_token_type_ids = torch.zeros_like(token_type_ids, device=self.device)
        return token_type_ids, ref_token_type_ids
        

    def construct_input_pos_ids(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return position_ids

    def construct_input_ref_pos_id_pair(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
        ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
        return position_ids, ref_position_ids
        

    def construct_attention_mask(self, input_ids):
        return torch.ones_like(input_ids)

