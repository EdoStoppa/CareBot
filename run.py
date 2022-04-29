import os
from sklearn.linear_model import LogisticRegression

import src.utils as utils
import src.train_and_test as ai
import src.bot_fsa as fsa

# Setting the name of the file containing the pre-trained word2vec representations
EMBEDDING_FILE = os.path.join("data", "w2v.pkl")


# model: A trained classification model
# word2vec: The pretrained Word2Vec model, if using other classification options (leave empty otherwise)
# Returns: This function does not return any values
#
# This function implements the main chatbot system --- it runs different
# dialogue states depending on rules governed by the internal dialogue
# management logic, with each state handling its own input/output and internal
# processing steps.
def run_chatbot(model, word2vec):

    # The first time simply go through a complete analysis
    fsa.welcome_state()
    fsa.get_info_state()
    fsa.health_check_state(model, word2vec, True)
    fsa.stylistic_analysis_state()

    next_state = fsa.check_next_state()

    while next_state != 'quit':
        if next_state == 'health_check':
            next_state = fsa.health_check_state(model, word2vec)
            continue
        if next_state == 'stylistic_analysis':
            next_state = fsa.stylistic_analysis_state()
            continue
        if next_state == 'check_next_state':
            next_state = fsa.check_next_state()
            continue
        # This case isn't really necessary, but why not be 100% sure?
        if next_state == 'quit':
            break

    print('\nThanks for talking with me!\nSee you again!\n')


if __name__ == "__main__":
    # Load the data needed to train the model
    lexicon, labels = utils.load_as_list(os.path.join("data", "dataset.csv"))

    # Load the Word2Vec representations
    word2vec = utils.load_w2v(EMBEDDING_FILE)

    # Instantiate and train the machine learning model
    logistic = LogisticRegression()
    logistic = ai.train_model(logistic, word2vec, lexicon, labels)

    # Reference code to run the chatbot
    run_chatbot(logistic, word2vec)
