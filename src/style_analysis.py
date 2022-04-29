import string
import nltk

# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of words in the input string.
def count_words(user_input):
    num_words = 0

    tkns = nltk.tokenize.word_tokenize(user_input)
    for tkn in tkns:
        if tkn not in string.punctuation:
            num_words += 1

    return num_words


# user_input: A string of arbitrary length
# Returns: A floating point value
#
# This function computes the average number of words per sentence
def words_per_sentence(user_input):
    wps = 0.0

    tot_words = 0
    sentences = nltk.tokenize.sent_tokenize(user_input)
    for sentence in sentences:
        tot_words += count_words(sentence)
    wps = tot_words / len(sentences)

    return wps


# user_input: A string of arbitrary length
# Returns: A list of (token, POS) tuples
#
# This function tags each token in the user_input with a Part of Speech (POS) tag from Penn Treebank.
def get_pos_tags(user_input):
    tagged_input = []

    tkns = nltk.word_tokenize(user_input)
    tagged_input = nltk.pos_tag(tkns)

    return tagged_input


# tagged_input: A list of (token, POS) tuples
# Returns: Seven integers, corresponding to the number of pronouns, personal
#          pronouns, articles, past tense verbs, future tense verbs,
#          prepositions, and negations in the tagged input
#
# This function counts the number of tokens corresponding to each of six POS tag
# groups, and returns those values.
def get_pos_categories(tagged_input):
    num_pronouns = 0
    num_prp = 0
    num_articles = 0
    num_past = 0
    num_future = 0
    num_prep = 0

    # Write your code here:
    num_pron_tags = ['PRP', 'PRP$', 'WP', 'WP$']
    num_prp_tags = ['PRP']
    num_art_tags = ['DT']
    num_past_tags = ['VBD', 'VBN']
    num_fut_tags = ['MD']
    num_prep_tags = ['IN']
    for (tkn, tag) in tagged_input:
        if tag in num_pron_tags: num_pronouns += 1
        if tag in num_prp_tags: num_prp += 1
        if tag in num_art_tags: num_articles += 1
        if tag in num_past_tags: num_past += 1
        if tag in num_fut_tags: num_future += 1
        if tag in num_prep_tags: num_prep += 1

    return num_pronouns, num_prp, num_articles, num_past, num_future, num_prep


# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of negation terms in a user input string
def count_negations(user_input):
    num_negations = 0

    neg = ["no", "not", "never", "n't"]
    tkns = nltk.word_tokenize(user_input)
    for tkn in tkns:
        if tkn.lower() in neg: num_negations += 1
    return num_negations


# num_words: An integer value
# wps: A floating point value
# num_pronouns: An integer value
# num_prp: An integer value
# num_articles: An integer value
# num_past: An integer value
# num_future: An integer value
# num_prep: An integer value
# num_negations: An integer value
# Returns: A list of three strings
#
# This function identifies the three most informative linguistic features from
# among the input feature values, and returns the psychological correlates for
# those features.
def summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations):
    informative_correlates = []

    # Creating a reference dictionary with keys = linguistic features, and values = psychological correlates.
    psychological_correlates = {}
    psychological_correlates["num_words"] = "Talkativeness, verbal fluency"
    psychological_correlates["wps"] = "Verbal fluency, cognitive complexity"
    psychological_correlates["num_pronouns"] = "Informal, personal"
    psychological_correlates["num_prp"] = "Personal, social"
    psychological_correlates["num_articles"] = "Use of concrete nouns, interest in objects/things"
    psychological_correlates["num_past"] = "Focused on the past"
    psychological_correlates["num_future"] = "Future and goal-oriented"
    psychological_correlates["num_prep"] = "Education, concern with precision"
    psychological_correlates["num_negations"] = "Inhibition"

    # Set thresholds
    num_words_threshold = 100
    wps_threshold = 20

    features = [(num_pronouns, 'num_pronouns'), (num_prp, 'num_prp'), (num_articles, 'num_articles'),
                (num_past, 'num_past'), (num_future, 'num_future'), (num_prep, 'num_prep'),
                (num_negations, 'num_negations')]

    if num_words > num_words_threshold: informative_correlates.append(psychological_correlates['num_words'])
    if wps > wps_threshold: informative_correlates.append(psychological_correlates['wps'])

    for _ in range(3):
        best = (-1, 'Nope')
        for elem in features:
            if elem[0] > best[0]: best = elem
        informative_correlates.append(psychological_correlates[best[1]])
        features.remove(best)

    informative_correlates = informative_correlates[:3]
    return informative_correlates