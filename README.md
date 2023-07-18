# Disaster Tweets Classification using Transformers

## 1. Introduction

Twitter has become an essential platform for real-time communication during emergencies, enabling individuals to share information about ongoing incidents. However, it is not always straightforward to determine if a tweet is genuinely indicating a disaster or if the language used is metaphorical or unrelated to actual emergencies. This ambiguity poses a challenge for organizations and agencies that rely on programmatic monitoring of Twitter to identify and respond to real emergencies promptly. This project aims to develop a machine-learning model that accurately predicts whether a tweet is about a real disaster or not.

## 2. Problem Description:

The problem at hand is to develop a machine-learning model that can accurately predict whether a given tweet is about a real disaster or not. 

## 3. Dataset Description

The project utilizes a [dataset](https://www.kaggle.com/competitions/nlp-getting-started/data) of 10,000 tweets that have been manually labeled as disaster-related or non-disaster-related. The dataset provides the ground truth for training and evaluating the model. The dataset contains the training and testing data. Each sample in the train and test set has the following information:

- The **text** of a tweet
- A **keyword** from that tweet (although this may be blank!)
- The **location** the tweet was sent from (may also be blank)

The columns details are as follows:
- **id** - a unique identifier for each tweet
- **text** - the text of the tweet
- **location** - the location the tweet was sent from (may be blank)
- **keyword** - a particular keyword from the tweet (may be blank)
- **target** - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

## 4. Exploratory Data Analysis (EDA)

During the EDA phase, the dataset is analyzed to gain insights and understanding. This includes assessing the distribution of tweet lengths, examining the balance of the target labels, and exploring any other relevant patterns or characteristics of the data.

#### 4.1 Heatmap of training data with NULL values

<img src="https://github.com/MUmairAB/Disaster-Tweets-Classification-using-Transformers/blob/main/Images/Heatmap%20of%20training%20data%20before%20discarding%20NULL%20values.png?raw=true" style="height: 433px; width:529px;"/>

#### 4.2 Heatmap of test data with NULL values

<img src="https://github.com/MUmairAB/Disaster-Tweets-Classification-using-Transformers/blob/main/Images/Heatmap%20of%20test%20data%20before%20discarding%20NULL%20values.png?raw=true" style="height: 433px; width:529px;"/>

#### 4.3 Heatmap of training data after removing NULL values

<img src="https://github.com/MUmairAB/Disaster-Tweets-Classification-using-Transformers/blob/main/Images/Heatmap%20of%20training%20data%20after%20discarding%20NULL%20values.png?raw=true" style="height: 433px; width:529px;"/>

#### 4.4 Heatmap of test data after removing NULL values

<img src="https://github.com/MUmairAB/Disaster-Tweets-Classification-using-Transformers/blob/main/Images/Heatmap%20of%20test%20data%20after%20discarding%20NULL%20values.png?raw=true" style="height: 433px; width:529px;"/>

#### 4.5 Distribution of Target values

<img src="https://github.com/MUmairAB/Disaster-Tweets-Classification-using-Transformers/blob/main/Images/Distribution%20of%20Target%20values.png?raw=true" style="height: 453px; width:580px;"/>

#### 4.6 Top 30 keywords in tweets

<img src="https://github.com/MUmairAB/Disaster-Tweets-Classification-using-Transformers/blob/main/Images/Top%2030%20keywords%20in%20tweets.png?raw=true" style="height: 404px; width:758px;"/>

#### 4.7 Top 30 Locations associated with tweets

<img src="https://github.com/MUmairAB/Disaster-Tweets-Classification-using-Transformers/blob/main/Images/Top%2030%20locations%20in%20tweets%20with%200%20discarded.png?raw=true" style="height: 416px; width:758px;"/>




## 5. Text Vectorization

To enable the model to process textual data, the tweets are converted into numerical representations through text vectorization techniques. This step ensures that the model can effectively learn from the text information.

## 6. Model Architecture: Transformer with Encoder-only Approach

The chosen model architecture is a Transformer model, specifically designed as an Encoder-only model. The Transformer architecture is well-suited for capturing contextual relationships and dependencies among words within tweets. The Encoder layer acts as a powerful feature extractor, learning representations that contribute to the tweet classification task.

#### 6.1 Positional Embedding layer
Incorporating a Positional Embedding layer, the model captures the sequential order of words in tweets. This positional encoding provides crucial information about word positions, allowing the model to understand the sequential structure and process the input text effectively.

#### 6.2 Model's plot

<img src="https://github.com/MUmairAB/Disaster-Tweets-Classification-using-Transformers/blob/main/Images/Model.png?raw=true" style="height: 663px, width: 582px"/>

#### 6.3 Model Summary
```
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, None)]            0         
                                                                 
 positional_embedding (Posit  (None, None, 256)        5162240   
 ionalEmbedding)                                                 
                                                                 
 transformer_encoder (Transf  (None, None, 256)        2121248   
 ormerEncoder)                                                   
                                                                 
 global_max_pooling1d_1 (Glo  (None, 256)              0         
 balMaxPooling1D)                                                
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_2 (Dense)             (None, 1)                 257       
                                                                 
=================================================================
Total params: 7,283,745
Trainable params: 7,283,745
Non-trainable params: 0
_________________________________________________________________
```

## 7. Model Training and Evaluation

The model is trained on the labeled dataset using **ADAM** optimization algorithms and **Binay Crossentropy**loss function. The performance metric of accuracy is computed to evaluate the model's classification performance. Visualizations and additional evaluation techniques are employed to gain further insights into the model's effectiveness.

#### 7.1 Training and validation plot

<img src="https://github.com/MUmairAB/Disaster-Tweets-Classification-using-Transformers/blob/main/Images/training%20and%20validation%20loss.png?raw=true" style="width: 567px height: 453px"/>

#### 7.2 Training and validation accuracy

<img src="https://github.com/MUmairAB/Disaster-Tweets-Classification-using-Transformers/blob/main/Images/training%20and%20validation%20accuracy.png?raw=true" style="width: 567px height: 453px"/>

## 8. Deployment and Application

The trained model can be deployed to predict the disaster relevance of new, unseen tweets. This capability allows organizations, news agencies, and disaster relief organizations to programmatically monitor Twitter and identify tweets that require immediate attention and action.

## 9. Conclusion

In conclusion, this project presents a machine-learning solution for tweet classification, specifically determining the disaster relevance of tweets. By leveraging the power of Transformer models and incorporating NLP techniques, the developed model can accurately classify tweets as disaster-related or non-disaster-related. The project has practical applications in real-time emergency monitoring, assisting in efficient resource allocation and timely responses to critical situations. Further enhancements and optimizations can be explored to improve the model's performance and expand its utility in disaster management.
