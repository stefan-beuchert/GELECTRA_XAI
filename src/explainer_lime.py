from lime import lime_text
import numpy as np


class Lime_Explanation:

    def __init__(self, model, class_names, random_state = None, num_samples=500, num_features=8):
        self.model = model
        self.class_names = class_names
        self.num_samples = num_samples
        self.num_features = num_features

        self.random_state = random_state
        
        self.explanation = None


    def explain(self, question, context):

        lime_input = question + " " +  context

        # TODO. parameter richtig setzen 
        explainer = lime_text.LimeTextExplainer(class_names=self.class_names, random_state=self.random_state)

        return explainer.explain_instance(lime_input,
                                          self.prediction_function_qa,
                                          num_samples=self.num_samples,
                                          num_features=self.num_features, 
                                          top_labels=2)


    def prediction_function_qa(self, question_and_context_list):
        results = []
    
        for element in question_and_context_list:
            # Seperate question from context
            split = element.split("?")
            question = split[0]
            context = split[1]
            question = question + "?"
            
            all_tokens = []
            model_output,all_tokens = self.model.predict(question,context)
            predition_results = self.model.get_predicted_tokens(model_output, all_tokens)
            
            start_token_prob = predition_results[0][1]
            end_token_prob = predition_results[1][1]

            row = []
            row.append(start_token_prob)
            row.append(end_token_prob)
            results.append(row)
        
        results_array = np.array(results)
        return results_array