from sentence_suggestion import inference

inference.SearchSpace(
     input_vocab_path='C:\\Users\\under\\GitHub\\Langauge_model\\sentence_suggestion\\inference\\data\\MASKED_TEXT.Field',
     output_vocab_path='C:\\Users\\under\\GitHub\\Langauge_model\\sentence_suggestion\\inference\\data\\TARGET_TEXT.Field',
     device_type='cpu',
     sentence_candidates_path='C:\\Users\\under\\GitHub\\Langauge_model\\sentence_suggestion\\inference\\data\\space_sample.csv'
)
