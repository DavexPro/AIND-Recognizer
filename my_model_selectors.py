import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        bic_scores = []
        """
        N is the number of data points, f is the number of features:

        N, f = self.X.shape

        Having m as the num_components, The free parameters p are a sum of:

        The free transition probability parameters, which is the size of the transmat matrix less one row because they add up to 1 and therefore the final row is deterministic, so m*(m-1)
        The free starting probabilities, which is the size of startprob minus 1 because it adds to 1.0 and last one can be calculated so m-1
        The number of means, which is m*f
        Number of covariances which is the size of the covars matrix, which for "diag" is m*f
        All of the above is equal to:

        p = m^2 +2mf-1

        Finally, the BIC equation is:

        BIC = -2 * logL + p * logN
        """
        try:
            n_components = range(self.min_n_components, self.max_n_components + 1)
            for num_states in n_components:
                model = self.base_model(num_states)
                log_l = model.score(self.X, self.lengths)
                p = num_states ** 2 + 2 * num_states * model.n_features - 1
                bic_score = -2 * log_l + p * math.log(num_states)
                bic_scores.append(bic_score)
        except Exception as e:
            pass

        states = n_components[np.argmin(bic_scores)] if bic_scores else self.n_constant
        return self.base_model(states)


class SelectorDIC(ModelSelector):
    ''' 
    Abbr.
        - DIC - Discriminative Information Criterion

    Equation.
        - DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models = []
        dic_scores = []
        other_words = []

        for word in self.words:
            if word != self.this_word:
                other_words.append(self.hwords[word])
        try:
            for num_states in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(num_states)
                word_log_p = model.score(self.X, self.lengths)
                models.append([word_log_p, model])

        except Exception as e:
            pass

        for model in models:
            word_log_p, hmm_model = model
            
            # equal to 1/(M-1)SUM(log(P(X(all but i))
            anti_log_p = np.mean([hmm_model.score(word[0], word[1]) for word in other_words])
            
            dic_score = word_log_p - anti_log_p 
            dic_scores.append([dic_score, hmm_model])

        best_dic = max(dic_scores, key = lambda x: x[0])[1] if dic_scores else None
        return best_dic


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        cv_scores = []
        kf = KFold(n_splits = 3, shuffle = False, random_state = None)

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            log_ls = []
            try:
                if len(self.sequences) > 2:
                    for train_index, test_index in kf.split(self.sequences):

                        self.X, self.lengths = combine_sequences(train_index, self.sequences)
                        X_test, lengths_test = combine_sequences(test_index, self.sequences)

                        hmm_model = self.base_model(num_states)
                        log_l = hmm_model.score(X_test, lengths_test)
                else:
                    hmm_model = self.base_model(num_states)
                    log_l = hmm_model.score(self.X, self.lengths)

                log_ls.append(log_l)
                cv_scores.append([np.mean(log_ls), hmm_model])

            except Exception as e:
                pass

        best_cv = max(cv_scores, key = lambda x: x[0])[1] if cv_scores else None
        return best_cv
