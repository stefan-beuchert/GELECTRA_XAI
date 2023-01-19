# GELECTRA
GELECTRA_MODEL_NAME = 'deepset/gelectra-base-germanquad'

# LIME
LIME_REGEX_FOR_TOKENIZATION = r'\W+'
LIME_CLASS_NAMES = ["Start-Token","End-Token"]

# list of possible NER tags
NER_tag_dict = {'' : 'Leerzeichen',
               'LOC' : 'Ortsangaben',
               'PER' : 'Personen',
               'MISC' : 'Diverses',
               'ORG' : 'Organisationen'}

NER_tag_list = list(NER_tag_dict.keys())

# list of possible POS tags
POS_tag_dict = {'ADJ' : 'Adjektive',
                'ADP' : 'Adposition', # oder auch Präposition im weitesten Sinne
                'PUNCT' : 'Satzzeichen',
                'ADV' : 'Adverb',
                'AUX' : 'Hilfsverben',
                'SYM' : 'Symbolen (#,€, ...)',
                'INTJ' : 'Ausrufen (psst, hurra, ...)',
                'CCONJ' : 'Bindeworten (und, oder, ...)',
                'X' : 'Anderes',
                'NOUN' : 'Nomen',
                'DET' : 'Bestimmungsworten',
                'PROPN' : 'Eigennamen',
                'NUM' : 'Zahlen',
                'VERB' : 'Verben',
                'PART' : 'Partikeln',
                'PRON' : 'Pronomen',
                'SCONJ' : 'Bindeworten (wenn, falls, während, ...)',
                'SPACE' : 'Leerzeichen',
                '' : 'Leerzeichen'}

POS_tag_list = list(POS_tag_dict.keys())