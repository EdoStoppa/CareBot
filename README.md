# *CareBot*
A simple Chatbot that performs a very basic medical and psychological evaluation on the user after interacting for a bit with the bot using Natural Language

## Chatbot components
I've mainly used this project to experiment with embeddings, NLP, and machine learning in general. To do that, this bot was divided into 3 main components:
the "medical evaluation" was performed using sentence embeddings and supervised learning, the "psychological evaluation" was performed by extracting some metrics
based on the POS tags of the sentences inserted by the user, and finally, the Finite State Automata that is used to manage the interaction between bot and user.

### "Medical evaluation" focus
The part of the project that I found most interesting was the "medical evaluation" part. To make it work we need two things: sentence embedding and a model that
can recognize if a person is feeling well or not. The first is achieved through multiple steps: First preprocessing the user input to clean it and split it into
single words, then obtaining the Word2Vec embeddings of all the words (using the pre-trained embeddings offered by Google), and in the end averaging the values of the
word embeddings to obtain the sentence embedding.

<br />
The second is obtained using supervised learning. Using a (very) small dataset of words and the associated labels, I trained a linear regression model to distinguish
an embedding related to good physical conditions from one related to bad physical conditions. I've tested other models (Support Vector Machine and Multi-layer
Perceptron) but the Linear Regression model outperformed the other two in every metric that I considered.

## Prerequisites
### Data
Almost everything is already present in the `data` folder, the only thing that's missing is the pickle file of the word embedding. The only thing needed is to extract 
`w2v.zip` in the same `data` folder. After that, you're completely set!

### Python Libraries
```
pandas
scikit-learn
nltk
```

## How to run
### Chatbot
Simply use this command
```python run.py```
The bot will start and you can interact with it without any other procedure.

### Model training and testing
If you're interested in running the comparison of the three algorithms that I considered using in this project, use the command<br />
```python src/train_and_test.py```
