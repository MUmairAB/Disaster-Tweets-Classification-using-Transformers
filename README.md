# Disaster Tweets Classification using Transformers

## 1. Introduction

Twitter has become an essential platform for real-time communication during emergencies, enabling individuals to share information about ongoing incidents. However, it is not always straightforward to determine if a tweet is genuinely indicating a disaster or if the language used is metaphorical or unrelated to actual emergencies. This ambiguity poses a challenge for organizations and agencies that rely on programmatic monitoring of Twitter to identify and respond to real emergencies promptly. This project aims to develop a machine learning model that accurately predicts whether a tweet is about a real disaster or not.

## 2. Problem Description:

The problem at hand is to develop a machine learning model that can accurately predict whether a given tweet is about a real disaster or not. 

## 3. Dataset Description

The project utilizes a [dataset](https://www.kaggle.com/competitions/nlp-getting-started/data) of 10,000 tweets that have been manually labeled as disaster-related or non-disaster-related. The dataset provides the ground truth for training and evaluating the model. The dataset contains the training and testing data. Each sample in the train and test set has the following information:

- The **text** of a tweet
- A **keyword** from that tweet (although this may be blank!)
- The **location** the tweet was sent from (may also be blank)

The columns details is a swloows:
- **id** - a unique identifier for each tweet
- **text** - the text of the tweet
- **location** - the location the tweet was sent from (may be blank)
- **keyword** - a particular keyword from the tweet (may be blank)
- **target** - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

## 4. Exploratory Data Analysis (EDA)

During the EDA phase, the dataset is analyzed to gain insights and understanding. This includes assessing the distribution of tweet lengths, examining the balance of the target labels, and exploring any other relevant patterns or characteristics of the data.


## 5. Text Vectorization

To enable the model to process textual data, the tweets are converted into numerical representations through text vectorization techniques. This step ensures that the model can effectively learn from the text information.

## 6. Model Architecture: Transformer with Encoder-only Approach

The chosen model architecture is a Transformer model, specifically designed as an Encoder-only model. The Transformer architecture is well-suited for capturing contextual relationships and dependencies among words within tweets. The Encoder layer acts as a powerful feature extractor, learning representations that contribute to the tweet classification task.

Incorporating a Positional Embedding layer, the model captures the sequential order of words in tweets. This positional encoding provides crucial information about word positions, allowing the model to understand the sequential structure and process the input text effectively.

## 7. Model Training and Evaluation

The model is trained on the labeled dataset using **ADAM** optimization algorithms and **Binay Crossentropy**loss function. Performance metrics of accuracy is computed to evaluate the model's classification performance. Visualizations and additional evaluation techniques are employed to gain further insights into the model's effectiveness.

## 8. Deployment and Application

Once trained and evaluated, the model can be deployed to predict the disaster relevance of new, unseen tweets. This capability allows organizations, news agencies, and disaster relief organizations to programmatically monitor Twitter and identify tweets that require immediate attention and action.

## 9. Conclusion
In conclusion, this project presents a machine-learning solution for tweet classification, specifically determining the disaster relevance of tweets. By leveraging the power of Transformer models and incorporating NLP techniques, the developed model can accurately classify tweets as disaster-related or non-disaster-related. The project has practical applications in real-time emergency monitoring, assisting in efficient resource allocation and timely responses to critical situations. Further enhancements and optimizations can be explored to improve the model's performance and expand its utility in disaster management.
