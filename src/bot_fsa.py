import re
import src.utils as utils
import src.style_analysis as style

# user_input: A string of arbitrary length
# Returns: Two strings (a name, and a date of birth formatted as MM/DD/YY)
#
# This function extracts a name and date of birth, if available, from an input
# string using regular expressions.  Names are assumed to be UTF-8 strings of
# 2-4 consecutive camel case tokens, and dates of birth are assumed to be
# formatted as MM/DD/YY.  If a name or a date of birth can not be found in the
# string, return an empty string ("") in its place.
def extract_user_info(user_input):
    name = ""
    dob = ""

    name_re = re.compile(r"( |^)[A-Z][a-zA-Z.\-&']*( [A-Z][A-Za-z.\-&']*){1,3}( |$)")
    dob_re = re.compile(r"( |^)(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/[0-9][0-9]( |$)")

    name = name_re.search(user_input)
    dob = dob_re.search(user_input)
    if (name != None):
        name = name.group()
        if (name[0] == ' '):  name = name[1:]
        if (name[-1] == ' '): name = name[:-1]
    else:
        name = ''
    if (dob != None):
        dob = dob.group()
        if (dob[0] == ' '):  dob = dob[1:]
        if (dob[-1] == ' '): dob = dob[:-1]
    else:
        dob = ''

    return name, dob

# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements the chatbot's welcome states.  Feel free to customize
# the welcome message!  In this state, the chatbot greets the user.
def welcome_state():
    # Display a welcome message to the user
    # *** Replace the line below with your updated welcome message from Project Part 1 ***
    careBot = '   _____               ____        _   \n' \
              + '  / ____|             |  _ \      | |  \n' \
              + ' | |     __ _ _ __ ___| |_) | ___ | |_ \n' \
              + ' | |    / _` | \'__/ _ \  _ < / _ \| __|\n' \
              + ' | |___| (_| | | |  __/ |_) | (_) | |_ \n' \
              + '  \_____\__,_|_|  \___|____/ \___/ \__|\n' \

    print('\n' + careBot +
          "\nWelcome to the CareBot!\n" +
          "This chatbot is still a work-in-progress, and definetely it's not intended as a substitute for your doctor.\n" +
          "So please, if you need medical assistance call a real doctor!\n")

    return ""


# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that requests the user's name and date of
# birth, and then processes the user's response to extract that information.
def get_info_state():
    # Request the user's name and date of birth, and accept a user response of
    # arbitrary length
    user_input = input("What is your name and date of birth?\n" +
                       "Enter this information in the form: First Last MM/DD/YY\n")

    # Extract the user's name and date of birth
    name, dob = extract_user_info(user_input)
    while name == '' or dob == '':
        if name == '':
            print("I'm sorry, the format of your name is wrong. The correct format is: FirstName LastName.\n" +
                  'Please try again.\n')
        if dob == '':
            print("I'm sorry, the format of your date of birth is wrong. The correct format is: MM/DD/YY.\n" +
                  'Please try again.\n')

        user_input = input()
        name, dob = extract_user_info(user_input)

    print("Thanks {0}! I'll make a note that you were born on {1}\n".format(name.split()[0], dob))

    return ''


# model: The trained classification model used for predicting health status
# word2vec: The pretrained Word2Vec model
# first_time (bool): indicates whether the state is active for the first time.
# Returns: A string indicating the next state
#
# This function implements a state that asks the user to describe their health,
# and then processes their response to predict their current health status.
def health_check_state(model, word2vec, first_time=False):
    # Check the user's current health
    user_input = input("How are you feeling today?\n")

    while len(user_input) == 0:
        user_input = input("I'm sorry, I didn't understand that.\n" +
                           "Can you repeat?\n")

    # Predict whether the user is healthy or unhealthy
    w2v_test = utils.string2vec(word2vec, user_input)
    label = model.predict(w2v_test.reshape(1, -1))
    if label == 0:
        print("Great! It sounds like you're healthy.\n")
    elif label == 1:
        print("Oh no! It sounds like you're unhealthy.\n")
    else:
        print("Hmm, that's weird. My classifier predicted a value of: {0}\n".format(label))

    out = 'stylistic_analysis' if first_time else "check_next_state"
    return out


# This function does not take any arguments
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what's on their mind, and
# then analyzes their response to identify informative linguistic correlates to
# psychological status.
def stylistic_analysis_state():
    user_input = input("I'd also like to do an informal psychological analysis.\nWhat's on your mind today?\n")

    while len(user_input) < 20:
        if len(user_input) == 0:
            user_input = input("I'm sorry, I didn't understand that.\n" +
                               "Can you repeat?\n")
        if len(user_input) > 0:
            user_input = input("I'm sorry, can you give me more details?\n" +
                               "Please use longer/multiple sentences, otherwise I won't" +
                               "be able to analyze your style!\n")

    num_words = style.count_words(user_input)
    wps = style.words_per_sentence(user_input)
    pos_tags = style.get_pos_tags(user_input)
    num_pronouns, num_prp, num_articles, num_past, num_future, num_prep = style.get_pos_categories(pos_tags)
    num_negations = style.count_negations(user_input)

    # Generate a stylistic analysis of the user's input
    informative_correlates = style.summarize_analysis(num_words, wps, num_pronouns,
                                                num_prp, num_articles, num_past,
                                                num_future, num_prep, num_negations)
    print("Thanks! Based on my stylistic analysis, I've identified the following psychological correlates in your response:")
    for correlate in informative_correlates:
        print("- {0}".format(correlate))
    print()

    return "check_next_state"


# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that checks to see what the user would like
# to do next.
def check_next_state():
    next_state = ""

    in_user = input('How can I help you now?\n' +
                    '   a) Quit the conversation\n' +
                    '   b) Redo the health check\n' +
                    '   c) Redo the stylistic analysis\n').lower()


    for _ in range(5):
        # Force the user to input something
        if len(in_user) == 0:
            in_user = input("\nI'm sorry, but you need to say something for me to understand you.\n" +
                            'How can I help you now?\n' +
                            '   a) Quit the conversation\n' +
                            '   b) Redo the health check\n' +
                            '   c) Redo the stylistic analysis\n').lower()

        # If the user uses the letter associated to the questions, use simpler logic
        if len(in_user) == 1:
            if in_user in ['a', 'b', 'c']:
                if in_user == 'a':
                    next_state = 'quit'
                    break
                if in_user == 'b':
                    next_state = 'health_check'
                    break
                if in_user == 'c':
                    next_state = 'stylistic_analysis'
                    break
            else:
                in_user = input("\nI'm sorry, but I didn't understand that.\n" +
                                "Can you rephrase what have you just said?\n").lower()

        if len(in_user) > 1:
            # Terrible logic to recognize negative sentences (not cool, but it works well in practice)
            reg_a = re.compile(r"((n\'t|don\'t|not).+(continue|repeat))|(quit|terminate|exit|end|bye)")
            reg_a_neg = re.compile(r"(n\'t|don\'t|not).+(quit|terminate|exit|end|bye)")
            reg_b = re.compile(r"(((^|.* )(re)?(do|take) )?.*(health([ -]?check| analysis)))")
            reg_b_neg = re.compile(r"((n\'t|don\'t|not).+ (re)?(do|take) .*(health([ -]?check| analysis)))")
            reg_c = re.compile(r"(((^|.* )(re)?(do|take) )?.*(stylistic (check|analysis)))")
            reg_c_neg = re.compile(r"((n\'t|don\'t|not).+ (re)?(do|take) .*(stylistic (check|analysis)))")

            a, a_neg = reg_a.search(in_user), reg_a_neg.search(in_user)
            b, b_neg = reg_b.search(in_user), reg_b_neg.search(in_user)
            c, c_neg = reg_c.search(in_user), reg_c_neg.search(in_user)

            is_a = a is not None and b is None and c is None
            is_b = a is None and b is not None and c is None
            is_c = a is None and b is None and c is not None

            if is_a:
                if a_neg is None:
                    next_state = "quit"
                    break
            if is_b:
                if b_neg is None:
                    next_state = "health_check"
                    break
            if is_c:
                if c_neg is None:
                    next_state = "stylistic_analysis"
                    break

            in_user = input("\nI'm sorry, but I didn't understand that.\n" +
                            "Can you rephrase what have you just said?\n").lower()


    if next_state == '':
        next_state = 'quit'
        print("\nI'm sorry, but today I'm quite slow.\n" +
              "That's not your fault, it's just that sometimes it happens.\n" +
              "I need to fix myself a little, then we can talk again!\n")

    return next_state
