import os
import src.utils as utils
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# Setting the name of the file containing the pre-trained word2vec representations
EMBEDDING_FILE = os.path.join("data", "w2v.pkl")

# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# training_data: A list of training documents
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model(model, word2vec, training_documents, training_labels):
    # Generate a list of embeddings
    doc_embeddings = []
    for doc in training_documents:
        doc_embeddings.append(utils.string2vec(word2vec, doc))

    # Train the model
    model.fit(doc_embeddings, training_labels)

    return model

# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model(model, word2vec, test_documents, test_labels):
    # Generate a list of embeddings
    doc_embeddings = []
    for doc in test_documents:
        doc_embeddings.append(utils.string2vec(word2vec, doc))

    # Obtain a prediction for all test data
    predicted_labels = model.predict(doc_embeddings)

    # Calculate all the metrics
    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    accuracy = accuracy_score(test_labels, predicted_labels)

    return (precision, recall, f1, accuracy)

def analyze_models():
    # Load the dataset
    lexicon, labels = utils.load_as_list("dataset.csv")
    test_data, test_labels = utils.load_as_list("test.csv")

    # Load the Word2Vec representations
    word2vec = utils.load_w2v(EMBEDDING_FILE)

    # Instantiate the models
    names = ['Logistic Regression', 'Support Vector Machine', 'Multi-Layer Perceptron']
    models = [LogisticRegression(), LinearSVC(), MLPClassifier(max_iter=5000)]
    
    # Training models
    print('\nStarted Training!\n')
    for idx in range(len(models)):
        print(f'Training {names[idx]}...')
        models[idx] = train_model(models[idx], word2vec, lexicon, labels)
    print('\n--------------------------------------------------------\n')

    # Test models
    print('Started Testing!\n')
    results = []
    for name, model in zip(names, models):
        print(f'Training {name}...')
        results.append(test_model(model, word2vec, test_data, test_labels))
    print('\n--------------------------------------------------------\n')

    # Printing the results (Terrible to write, but looks good as output)
    print('\t\t\t\tPrecision\tRecall\t     Accuracy\t     F1')
    for name, res in zip(names, results):
        print(f'{name}\t\t   '+
              f'{res[0]:.2f}\t\t '+
              f'{res[1]:.2f}\t       '+
              f'{res[2]:.2f}\t    '+
              f'{res[3]:.2f}')

if __name__ == '__main__':
    analyze_models()

