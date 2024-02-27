# ************ Step 1 - Import the required library ************ 


# Import the required spaCy library
# We will be using the spaCy model 
# For the sentiment analysis task
# Load the English pipeline-trained model

import spacy  # importing spacy
from spacytextblob.spacytextblob import SpacyTextBlob
nlp = spacy.load('en_core_web_sm')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

# Load data set, input the source of data "amazon_product_reviews.csv".
df = pd.read_csv("amazon_product_reviews.csv")

# ************ Step 2 - Preprocessing the text data ************ 

# Removing the missing values in the 'review.text' column
# Using the dropna() function from Pandas
clean_data = df.dropna(subset=['reviews.text'])
print(clean_data['reviews.text'])

# select the 'review.text' column from the dataset and retrieve its data
# Then put it into a list
review_list = clean_data['reviews.text'].to_list()


# Create an empty list for cleaning data.
filtered_words =[]
clean_review_list = []

# Removing stop words with NLTK from the review texts
stop_words = set(stopwords.words('english'))

# Remove the stop words (.is_stop attribute in spaCy)
# and punctuation into the text (.is_punct attribute in spaCy)
# For data cleaning
# employ the filtered word of tokens or words(words with no stop words) 
# for conducting sentiment analysis.

for review_sentence in review_list:
    # Process the text using spaCy
    doc = nlp(review_sentence)
    # Remove stopwords
    filtered_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    # Join the filtered words to form a clean text
    clean_text = ' '.join(filtered_words)
    # use .lower(), .strip(), str() to further cleaning the texts
    clean_text = str(clean_text.lower()).strip()
    clean_review_list.append(clean_text) 

    
# Create a Data Frame using Panda for the Clean Review.
    
df_clean_reviews = pd.DataFrame(clean_review_list)
df_clean_reviews = df_clean_reviews.rename(columns={0: 'Clean Review'})
print(df_clean_reviews)



# ************ Step 3 - Performing sentiment analysis ************ 


# Function for sentiment analysis
# Takes in review as an input and predicts sentiment.

nlp.add_pipe('spacytextblob')


def clean_review(review):
    doc = nlp(review)
    polarity = doc._.blob.polarity
    sentiment = doc._.blob.sentiment
    print(polarity, sentiment)

clean_review((clean_review_list[0]))


# Further sentiment studies on determining whether it expresses a positive, negative, or neutral sentiment.
# Use the use the spaCy model and the .sentiment and .polarity attributes
# Checking polarity on review.
polarity_reviews = []

for review in clean_review_list:
    doc = nlp(review)
    polarity = doc._.blob.polarity
    polarity_reviews.append(polarity)

print(polarity_reviews)


# Create a Data Frame using Panda for the Polarity of the Clean Reviews.

df_polarity_reviews = pd.DataFrame(polarity_reviews)
df_polarity_reviews = df_polarity_reviews.rename(columns={0: 'Polarity'})
print(df_polarity_reviews)


# Reset the index for the data frame for a clean review 
# to prepare it to join with the data frame of Polarity of the Clean Reviews.
df_clean_reviews = df_clean_reviews.reset_index()
df_clean_reviews = df_clean_reviews.rename(columns={"index":"New_ID"})
df_clean_reviews['New_ID'] = df.index

print(df_clean_reviews)


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

# Run the polarity score on the entire dataset
print(df_clean_reviews)
res ={}

for i, row in tqdm(df_clean_reviews.iterrows(), total=len(df_clean_reviews)):
    review_all = row['Clean Review']
    myid = row['New_ID']
    res[myid] = sia.polarity_scores(review_all)


# Print out the positive, negative, or neutral sentiment for each review.
print(res)

# Create DataFrame of the positive, negative, or neutral sentiment.
vadars = pd.DataFrame(res).T
print(vadars)

# Reset the index for the data frame for the positive, negative, or neutral sentiment.
# to prepare it to join with the data frame of Clean Reviews.
vadars = vadars.reset_index()
vadars = vadars.rename(columns={"index":"New_ID"})
vadars['New_ID'] = df.index
print(vadars)

# Merging the data frame for the positive, negative, or neutral sentiment and Clean reviews.
new_dataframe = df_clean_reviews.merge(vadars, how='right')

print(new_dataframe)

# Reset the index for the data frame for the orginal data frame with fewer columns.
original_df_enhanced = df[['name', 'reviews.doRecommend', 'reviews.rating']]

original_df_enhanced = original_df_enhanced.reset_index()
original_df_enhanced = original_df_enhanced.rename(columns={"index":"New_ID"})
original_df_enhanced['New_ID'] = df.index

# Merging data set for final data frame.
merge_all_dataframe = original_df_enhanced.merge(new_dataframe, how='right')

print(merge_all_dataframe)



# ************ Step 4 - compare the similarity of two product reviews ************ 


# Choose two product reviews from the 'review.text' column with indexing (index 3 & 7)
first_review_compare = clean_data['reviews.text'][3]
Second_review_compare = clean_data['reviews.text'][7]

# function to compare the similarity of two product reviews.
def compare_similarity():
        similarity = nlp(first_review_compare).similarity(nlp(Second_review_compare))
        print(similarity)
        print('The Review Text Index 3 is: ' + clean_data['reviews.text'][3])
        print('The Review Text Index 7 is: '+ clean_data['reviews.text'][7])
        print('The similarity score between Review Text Index 1 and Review Text Index 3 is ' + str(similarity))

compare_similarity()



# ************ Step 5 - Testing the model with some sample review ************ 

print('Testing the model with some sample review: ')
print('The Review Text Index 3 is: ' + clean_data['reviews.text'][3])
clean_review((clean_data['reviews.text'][3]))

print('The Review Text Index 7 is: '+ clean_data['reviews.text'][7])
clean_review((clean_data['reviews.text'][7]))

# Since Review [3] in words is positive. The polarity score is higher than 0 (0.240497)
# While Review [7] in words is negative. The polarity score is less than 0 (-0.2)
# The testing of the model shows that the model is fairly accurate.



# ************ Step 6 - Data Visualisation for sentiment analysis results ************ 


# Plot sentiment analysis results
'''# Barplot based on compound score and Amazon review ratings

ax = sns.barplot(data=merge_all_dataframe, x= 'reviews.rating', y= 'compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()
'''
# Barplot based on positive, neutral, negative scores and Amazon review ratings
fig, axs = plt.subplots(1,3,figsize=(15,3))
sns.barplot(data=merge_all_dataframe, x= 'reviews.rating', y= 'pos', ax=axs[0])
sns.barplot(data=merge_all_dataframe, x= 'reviews.rating', y= 'neu', ax=axs[1])
sns.barplot(data=merge_all_dataframe, x= 'reviews.rating', y= 'neg', ax=axs[2])

axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negective')
plt.tight_layout()
plt.show()

# Project inspired by https://www.youtube.com/watch?v=QpzMWQvxXWk
# Python Sentiment Analysis Project with NLTK and Transformers. Classify Amazon Reviews!!

# Things to improve: note:
# Exploring advanced techniques: 
# 1. Named Entity Recognition (NER)
# 2. sentiment analysis using deep learning models like BERT or GPT
# 3. topic modeling with algorithms: Latent Dirichlet Allocation (LDA)

# ************ End of code ************
