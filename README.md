
Fake News Classifier:
By: Pinky Chauhan (University of Illinois at Urbana Champaign)

Overview:
This objective of this project is to build a classifier system based on machine learning that is able to identify fake news from real/reliable news given a news title and/or news text content as the input. Such a tool can be integrated with social media platforms to flag potentially fake articles or filter those out.
This is essentially a data categorization problem where I have trained several classifier models on the following dataset from Kaggle:
https://www.kaggle.com/c/fake-news/data
After evaluation based on various performance metrics, one of the models (in this case, Linear SVC over unigram bag-of-words/TF-IDF representation) is integrated in the final tester notebook to test with news data.
The classifier takes a news article (title and text) as input and provides a prediction for the news article as either of the 2 categories:
-	Fake News
-	Reliable News

Final Project report:
Presentation:
