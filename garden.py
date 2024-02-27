# ======= Working with the spaCy ===== #
# Import the required spaCy library
# Load the English pipeline-trained model


import spacy
nlp = spacy.load('en_core_web_sm')

# Put 2 garden path sentences from the web.
# Store the 2 garden sentences into a list as 'gardenpathSentences'
gardenpathSentences = ['The complex houses married and single soldiers and their families.', 'While I was surfing Reddit went down.']

# Add 3 required sentences to the same list.
extension = ['Mary gave the child a Band-Aid.', 'That Jill is never here hurts.', 'The cotton clothing is made of grows in Mississippi.']
for sent in extension:
    gardenpathSentences.append(sent)

# Tokenisation------------------------------------------------------------------

# Tokenizing the ext
# Avoid returning tokens that are punctuation or white space


for content in gardenpathSentences:
    doc = nlp(content)
    doc.text.split()
    print([(token, token.orth_, token.orth) for token in doc if not token.is_punct | token.is_space])

# Output stopwords (Additional task)
    
list_stopwords =[]
for content in gardenpathSentences:
    doc = nlp(content)
    for word in doc:
        if word.is_stop==True:
            list_stopwords.append(word)
print(list_stopwords)



# Lemmatization------------------------------------------------------------------  (Additional task)


# To reduce the word to its basic form

for content in gardenpathSentences:
    nlp_practice = nlp(content)
    for word in nlp_practice:
        print(word.text,word.lemma_)



# Named entity recognition------------------------------------------------------------------
    

# Part of Speech (POS) Tagging (Additional task)
for content in gardenpathSentences:
    doc = nlp(content)
    for word in doc:
        print(word.text,word.pos_)
 
# Get an explanation of an entity and print it (Additional task)
entity_det = spacy.explain("DET")
print(f"DET:{entity_det}")

entity_adj = spacy.explain("ADJ")
print(f"ADJ:{entity_adj}")

entity_noun = spacy.explain("NOUN")
print(f"NOUN:{entity_noun}")

entity_punct = spacy.explain("PUNCT")
print(f"PUNCT:{entity_punct}")

entity_cconj = spacy.explain("CCONJ")
print(f"CCONJ:{entity_cconj}")

entity_pron = spacy.explain("PRON")
print(f"PRON:{entity_pron}")

entity_propn = spacy.explain("PROPN")
print(f"PROPN:{entity_propn}")

entity_verb = spacy.explain("VERB")
print(f"VERB:{entity_verb}")

entity_sconj = spacy.explain("SCONJ")
print(f"SCONJ:{entity_sconj}")

entity_aux = spacy.explain("AUX")
print(f"AUX:{entity_aux}")

entity_adp = spacy.explain("ADP")
print(f"ADP:{entity_adp}")



# Get labels and entities and print them

from spacy import displacy

for content in gardenpathSentences:
    nlp_content = nlp(content)
    print([(i, i.label_, i.label) for i in nlp_content.ents])


# print the meaning of entities

entity_norp = spacy.explain("NORP")
print(f"NORP:{entity_norp}")

entity_gpe = spacy.explain("GPE")
print(f"GPE:{entity_gpe}")



# Comment about two entities that I looked up------------------------------------------------------------------


'''The entity and its explanation that you looked up are:
1. Reddit, 'NORP'
2. Mississippi, 'GPE'
from the garden sentence: 
'While I was surfing Reddit went down.'
'The cotton clothing is made of grows in Mississippi.'''

''' For 1. Reddit, 'NORP':
'Reddit' in 'While I was surfing Reddit went down.' is a proper noun.' 
While Reddit is a social forum organisation founded in the US,
named entity recognition identified it as -
NORP: Nationalities or religious or political groups
which is somewhat close but not exactly.
It seems to fail to recognise the true name and purpose of this group with the trained model of spaCy.
The entity 'Reddit' does not exactly make sense in terms of the word associated with it.
'''

''' For 2.  Mississippi, 'GPE':
'Mississippi' in 'The cotton clothing is made of grows in Mississippi.'. 
Mississippi is a known state in the US. 
named entity recognition identified it as -
GPE: Countries, cities, states
which have got it spot on State from 'GPE'
The trained model of spaCy for the word 'Mississippi' is accurate.
the entity 'Mississippi' makes sense in terms of the word associated with it.
'''

'''
As a whole, while named entity recognition from spaCy tells an unstructured name string 
and categorised them into standard forms. 
While a garden-path sentence usually makes the reader confused with a word to be a noun (usually common) and a verb.
it will not affect the name entity recognition process.
As name entity recognition usually recognises proper nouns, or dates and collective nouns,
the word for that proper noun identified (Reddit, 'NORP' and Mississippi, 'GPE')
remain the same meaning in these garden sentences.
Therefore the entity (proper noun) in general makes sense in terms of the word associated with it'''
