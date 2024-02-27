# Part 1 - Find similarity between words.

# ======= Working with the spaCy ===== #
# Import the required spaCy library
# Load the English pipeline-trained model


print('\n')
print("-------------Using medium size model---------------")


import spacy  # importing spacy
nlp = spacy.load('en_core_web_md') # Load the English pipeline-trained model (medium size)

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


# Note what you noticed about the similarities between cat, monkey and banana in the handout.
# Using medium size model
# cat monkey 0.5929929614067078, cat banana 0.2235882729291916, banana monkey 0.404150128364563
# As Cats and Monkey are both animals, they score the highest in terms of similarity
# Monkeys usually eat bananas so monkeys and bananas have similarities and the score is fairly high.
# Cat and Banana have no relation as Cats are animals and Banana is fruit.
# Cat also did not eat Banana as such so the similarity is pretty weak
# So it reflects on the lower score for Cat and Banana




print('\n')
print("-------------Using small size model---------------")


import spacy
nlp = spacy.load('en_core_web_sm') # Load the English pipeline-trained model (small size)


tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


# Note what you noticed about the similarities between cat, monkey and banana in the handout.
# Using small size model
# cat monkey 0.6455236673355103, cat banana 0.22147203981876373, banana monkey 0.4232020974159241
# As Cats and Monkey are both animals, they score the highest in terms of similarity
# Monkeys usually eat bananas so monkeys and bananas have similarities and the score is fairly high.
# Cat and Banana have no relation as Cats are animals and Banana is fruit.
# Cat also did not eat Banana as such so the the similarity is pretty weak
# So it reflects on the lower score for Cat and Banana
# Output shows a similar conclusion for both small and medium models.




print('\n')
print("---------------Example of your own-----------------")
print('\n')
print("-------------Using medium size model---------------")

nlp = spacy.load('en_core_web_md') # Load the English pipeline-trained model (medium size)

tokens_1 = nlp('diamond paper gold pencil ')
for token1 in tokens_1:
    for token2 in tokens_1:
        print(token1.text, token2.text, token1.similarity(token2))

# diamond paper 0.26512962579727173, diamond gold 0.6683288812637329, diamond pencil 0.3247971832752228, paper pencil 0.555428147315979, gold pencil 0.222731813788414, paper gold 0.6583979725837708
# Diamond and Gold are both precious and both are metal and with decorative use.
# They have significant similarities hence their high similarity score.
# Paper and Pencils are Stationary so they have quite a strong relationship in terms of their category.
# Diamond and Pencil, Diamond and Paper have less similarity. 
# With Diamond and Pencil slightly higher because they can both be used as drilling tools 
# Gold and Pencil, Gold and Paper have less similarity. One being precious metal and the other stationary. 



print('\n')
print("-------------Using small size model---------------")

nlp = spacy.load('en_core_web_sm') # Load the English pipeline-trained model (small size)

tokens_1 = nlp('diamond paper gold pencil ')
for token1 in tokens_1:
    for token2 in tokens_1:
        print(token1.text, token2.text, token1.similarity(token2))

# diamond paper 0.5651285648345947, diamond gold 0.5855413675308228, diamond pencil 0.20398299396038055, paper pencil 0.2830146849155426, pencil gold 0.28354060649871826, paper gold 0.6583979725837708
# Diamond and Gold are both precious and both are metal and with decorative use.
# They have significant similarities hence their high similarity score.
# Diamond and Pencil, Gold and Pencil have very low similarity. One being precious metal and the other stationary.
# Hence their very low score.
# Diamond and Pencil vs Gold and Pencil do not pick up the fact that Diamond and Pencil are drilling tools so should have a higher score.
# Diamond and Paper, Gold and Paper have almost no similarity.
# But they have a surprisingly high score, which shows the small model is not too accurate. 
# Also Paper and Pencil should have a relatively high similarity but they have a very low similarity score.
# Which shows the inaccuracy of a small model.
# In conclusion, to pick a model for semantic similarity in spaCy.
# It is better to use a larger model for comparison.

print('\n')


# Part 2 - Find a movie recommandation based on reviews.


import spacy  # importing spacy
nlp = spacy.load('en_core_web_md') # Load the English pipeline-trained model (medium size)

planet_hulk = 'Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk lands on the planet Sakaar where he is sold into slavery and trained as a gladiator.'

print('\n')
print("-------------T20 - Practical Task 2---------------")
print('\n')
print("----------------Movies similarity-----------------")

file = open("movies.txt", "r")

# read the text file by lines using readlins().
# Store the text into a list 'sentences'.

sentences = file.readlines()

# Create an empty list to store the similarity score for each movie
list_of_similarity =[]

# compare to the 'Planet Hulk' movie.
# Create a function 'compare_similarity' 
# Take the description of the movie 'Planet Hulk' as a parameter.
# return the title of the most similar movie.

def compare_similarity(planet_hulk):
    for sentence in sentences:
        similarity = nlp(sentence[9:]).similarity(nlp(planet_hulk))
        list_of_similarity.append(similarity)
    print('The title of the most similar movie is ' + (str(sentences[list_of_similarity.index(max(list_of_similarity))])[:7]) + '.')

compare_similarity(planet_hulk)
