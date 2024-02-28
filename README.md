# Final Captone - NLP Application - Sentiment Analysis.py

## Project name - NLP Application - Sentiment Analysis

## Summary of the project - the NLP application: Sentiment analysis - 
This is a Python programme for carrying out sentiment analysis for Amazon product reviews taken Kaggle dataset website.

## Table of content - 
###### Step 1. A description of the dataset used.
The dataset that is being used is based on the website - Consumer Reviews of Amazon Products (kaggle.com) :
https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products .
Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products file by Datafiniti and saved to
amazon_product_reviews.csv
The data set has 5000 entries and has Amazon products ranging from tablets and Kindles that have around 24
columns with IDs.
We use this CSV file as a training set to show the review. ratings which is helpful for sentiment analysis.

###### Step 2. The pre-processing steps:
1. It removes the missing values in the ‘review. text’ column by using the dropna() function from Pandas.
2. It then selects the review.text column.
3. Tokenization is carried out to tokenize the text before it turns into a vector and removes unwanted tokens.
4. Remove stop words using the .is_stop attribute from spaCy.
5. Remove punctuation using the .is_punct attribute from spaCy.
6. .lower() to make text lowercase.
7. .strip() to remove white spaces.
8. str() to turn it into strings.
9. It then creates a clean list of texts.
10. Create a data frame of the clean text.
    
###### Step 3. Evaluation of results.
The result of the sentiment analysis has shown us a fairly accurate prediction. The Sentiment analysis predicts the
polarity of the reviews using TextBlob .polarity and .sentiment and outputs the positivity, negativity and neutral and its
compounds to a score. The positivity, negativity and neutral scores from the sentiment analysis match with the review
rating of the Amazon reviews.

###### Step 4. Insights into the model's strengths and limitations.
This project uses the Python nldk natural language toolkit. It shows fairly accurate results on a baseline level.
Strengths:
The NLP toolkit is open source and free with support on the NLP library. It is easy to use and produces quick and
mostly correct predictions of results. It uses Python.
Limitation:
However, there are some more complicated meanings of languages that the model did not pick up. In one of the
reviews it said ‘The product is a steal.’ (Review [4999]) The Model interpreted it as negative and did not pick up that
the meaning was positive. With the English language being a bit ambiguous at times, this basic traditional model
cannot pick up these more precise meanings and produce less accurate results. There are also multiple meanings
that this model shows are less efficient to use.

###### Step 5 - Testing the model with some sample review
Using .similarity to comparing the similarity of 2 reviews of the datasets.

###### Step 6 - Data Visualisation for sentiment analysis results
Using Bar chart to plot the results 



