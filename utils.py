import itertools
import os
from typing import Dict, Callable, Iterable, Tuple, List, Union

import nltk
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.sparse.csr import csr_matrix
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# import spacy
import re
from functools import lru_cache


# we may need to download nltk & spacy data locally...
if not os.path.isdir(f"{os.path.expanduser('~')}/nltk_data/corpora/stopwords"):
    nltk.download('stopwords')

if not os.path.isdir(f"{os.path.expanduser('~')}/nltk_data/corpora/punkt"):
    nltk.download('punkt')

__stemmer = PorterStemmer()


# ##################################################################
# AUXILIARY FUNCTIONS
# ##################################################################

def remove_nan_questions(x_train: pd.DataFrame, y_train: pd.DataFrame) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove those samples which contain a NaN in at least one question.

    Parameters
    ----------
    x_train: pd.DataFrame
        Two columns dataframe containing the feature questions
    y_train: pd.DataFrame
        One column dataframe containing the labels

    Returns
    -------
    dropped_x_train, dropped_y_train: Tuple[pd.DataFrame, pd.DataFrame]
        Dataframes without NaN in any sample

    """
    dropped_x_train = x_train.dropna(how="any")
    idx = set(x_train.index).intersection(dropped_x_train.index)
    dropped_y_train = y_train.loc[list(idx)]

    return dropped_x_train, dropped_y_train


# ##################################################################
# FUNCTIONS TO AGGREGATE FEATURE VECTORS
# ##################################################################

def _horizontal_stacking(x_q1: csr_matrix, x_q2: csr_matrix) -> csr_matrix:
    """
    Stack horizontally the 2 passed feature matrices

    Parameters
    ----------
    x_q1: csr_matrix
        Feature (sparse) matrix (each row is the feature vector
        obtained from the first question)
    x_q2: csr_matrix
        Feature (sparse) matrix of the second question

    Returns
    -------
    Feature (sparse) matrix with the questions merged

    """
    return hstack((x_q1, x_q2))


# TODO: the output did not fit in memory in either case: DEPRECATED
def _cosine_similarity(x_q1: csr_matrix, x_q2: csr_matrix) -> csr_matrix:
    # from sklearn.preprocessing import normalize
    # x_q1_normed = normalize(x_q1.tocsc(), axis=1)
    # x_q2_normed = normalize(x_q2.tocsc(), axis=1)
    # _cos_sim = x_q1_normed * x_q2_normed.T
    _cos_sim = x_q1.tocsr() * x_q2.tocsr().T
    # from sklearn.metrics.pairwise import cosine_similarity
    # _cos_sim = cosine_similarity(x_q1, x_q2, dense_output=False)
    return _cos_sim


def _abs_difference(x_q1: csr_matrix, x_q2: csr_matrix) -> csr_matrix:
    return np.abs(x_q1 - x_q2)


# ##################################################################
# FUNCTIONS TO PREPROCESS QUESTIONS
# ##################################################################

def _remove_punctuation(text: str):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        text = np.char.replace(text, symbols[i], ' ')
        text = np.char.replace(text, "  ", " ")
    text = np.char.replace(text, ',', '')
    return text


def _remove_stop_words(text: str, stop_words: Iterable[str]) -> str:
    return ' '.join([word for word in text.split() if word not in stop_words])


def _stemming(text: str) -> str:
    return " ".join([__stemmer.stem(w) for w in word_tokenize(text)])


def _to_british(text: str) -> str:
    text = re.sub(r"(...)our\b", r"\1or", text)
    text = re.sub(r"([bt])re\b", r"\1er", text)
    text = re.sub(r"([iy])s(e\b|ing|ation)", r"\1z\2", text)
    text = re.sub(r"ogue\b", "og", text)
    return text


# ##################################################################
# FUNCTIONS TO GENERATE EXTRA FEATURES
# ##################################################################


def _length_ratio(q1_w: List[str], q2_w: List[str]) -> float:
    """
    Return the question's length ratio (the first with respect the second).
    If any of them has 0 length (after preprocessing), the retrieved ratio
    is 0.

    Parameters
    ----------
    q1_w: List[str]
        List of words contained in the first (preprocessed) question's samples
    q2_w: List[str]
        List of words contained in the second (preprocessed) question's samples

    Returns
    -------
    ratio: float
        Question's length ratio

    """
    if any([len(_q) for _q in (q1_w, q2_w)]):
        return 0.
    return len(q1_w) / len(q2_w)


def _get_coincident_words_ratio(
        q1_w: List[str], q2_w: List[str]) -> float:
    """
    Count the ratio of coincident words with respect the total number of them.
    This is applied at a sample level.

    Parameters
    ----------
    q1_w: List[str]
        List of words contained in the first (preprocessed) question's samples
    q2_w: List[str]
        List of words contained in the second (preprocessed) question's samples

    Returns
    -------
    ratio: float
        Ratio of coincident words (between the 2 questions)

    """
    unique_q1, unique_q2 = set(q1_w), set(q2_w)
    return 2 * len(unique_q1 & unique_q2) / (len(unique_q1) + len(unique_q2))


def _coincident_keyword(
        q1_w: List[str], q2_w: List[str]) -> float:
    """
    Identify whether the 2 questions have coincident
    keywords and returns a normalized value proportional 
    to this ratio
    """
    keywords = {"What", "what", "Who", "who", "Which", "which", "Where",
                "where", "Why", "why", "When", "when", "How", "how", "Whose",
                "whose", "Can", "can"}
    unique_q1, unique_q2 = set(q1_w), set(q2_w)
    keywords_q1 = unique_q1 & keywords 
    keywords_q2 = unique_q2 & keywords
    
    if len(keywords_q1) == 0 and len(keywords_q2) == 0:
        denominator = 1
    else:
        denominator = (len(keywords_q1) + len(keywords_q2))
    return 2*len(keywords_q1 & keywords_q2)/denominator


def _jaccard_distance(
        q1_w: List[str], q2_w: List[str]) -> float:
    """
    Implement the Jaccard distance between two questions
    """
    unique_q1, unique_q2 = set(q1_w), set(q2_w)
    return 1 - len(unique_q1 & unique_q2) / len(unique_q1.union(unique_q2))


def _levenshtein_sim_w(
        q1_w: List[str], q2_w: List[str]) -> float:
    """
    Implements a modified Levenshtein Distance taking as elements
    all the words within the question
    """
    
    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):

        if s1 == len(q1_w) or s2 == len(q2_w):
            return len(q1_w) - s1 + len(q2_w) - s2

        # no change required
        if q1_w[s1] == q2_w[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),      # insert character
            min_dist(s1 + 1, s2),      # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)


# TODO: maximum recursion depth reached when calling the function: DEPRECATED
def _levenshtein_sim_l(
        q1_w: List[str], q2_w: List[str]) -> float:
    """
    Implements Regular Levenshtein Distance
    for all the letters from each word within the question
    """
    q1_w = " ".join(q1_w)
    q2_w = " ".join(q2_w)
    
    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):

        if s1 == len(q1_w) or s2 == len(q2_w):
            return len(q1_w) - s1 + len(q2_w) - s2

        # no change required
        if q1_w[s1] == q2_w[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),      # insert character
            min_dist(s1 + 1, s2),      # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)


# ##################################################################
# GLOBAL VARS CLASSES (defined extractors and aggregators)
# ##################################################################

_SUPPORTED_EXTRACTORS: dict = {
    'cv': CountVectorizer(
        ngram_range=(1, 1), lowercase=False),
    'cv_2w': CountVectorizer(
        ngram_range=(1, 2), lowercase=False),
    'tf_idf': TfidfVectorizer(ngram_range=(1, 1), lowercase=False),
    'tf_idf_2w': TfidfVectorizer(ngram_range=(1, 2), lowercase=False),
    # spacy: DISMISSED to reduce conda environment size and training loads
    # 'spacy_small': 'en_core_web_sm',
    # 'spacy_medium': 'en_core_web_md',
}

_SUPPORTED_AGGREGATORS: Dict[str, Callable] = {
    'stack': _horizontal_stacking,
    # 'cosine': _cosine_similarity,
    'absolute': _abs_difference,
}

_SUPPORTED_EXTRA_FEATURES: Dict[str, Callable] = {
    'coincident_ratio': _get_coincident_words_ratio,
    'coincident_keyword': _coincident_keyword,
    'jaccard': _jaccard_distance,
    'levenshtein_w': _levenshtein_sim_w,
    # 'levenshtein_l': _levenshtein_sim_l,
    'length_ratio': _length_ratio,
}


# PIPELINES CLASSES

class TextPreprocessor:
    def __init__(self,
                 remove_stop_words: bool = False,
                 remove_punctuation: bool = False,
                 to_lower: bool = False,
                 apply_stemming: bool = False,
                 british: bool = False) \
            -> None:
        self.remove_stop_words = remove_stop_words
        self.remove_punctuation = remove_punctuation
        self.to_lower = to_lower
        self.apply_stemming = apply_stemming
        self.british = british

        self.custom_stop_words: set = {
            "What", "what", "Who", "who", "Which",
            "which", "Where", "where", "Why", "why",
            "When", "when", "How", "how", "Whose", "whose", "Can"}
        self.stop_words = set(stopwords.words(
            'english')) - self.custom_stop_words

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        """
        This method takes a pandas DataFrame with two columns as input
        and performs several operations according to the initialization.
        These can be:
         - lower case transformation
        - removal of stop words
        - removal of punctuation signs

        before returning the preprocessed text in a new DataFrame.
        """
        df_prep = df.copy()

        # preprocessing
        if self.to_lower:
            df_prep['question1'] = df_prep['question1'].str.lower()
            df_prep['question2'] = df_prep['question2'].str.lower()
        # apply functions to the 'text' column of the DataFrame
        if self.remove_stop_words:
            df_prep.loc[:, 'question1'] = df_prep['question1'].apply(
                lambda _t: _remove_stop_words(_t, self.stop_words))
            df_prep.loc[:, 'question2'] = df_prep['question2'].apply(
                lambda _t: _remove_stop_words(_t, self.stop_words))
        if self.remove_punctuation:
            df_prep.loc[:, 'question1'] = df_prep['question1'].apply(
                lambda _t: _remove_punctuation(_t))
            df_prep.loc[:, 'question2'] = df_prep['question2'].apply(
                lambda _t: _remove_punctuation(_t))
        if self.british:
            df_prep.loc[:, 'question1'] = df_prep['question1'].apply(
                lambda _t: _to_british(_t))
            df_prep.loc[:, 'question2'] = df_prep['question2'].apply(
                lambda _t: _to_british(_t))
        if self.apply_stemming:
            df_prep.loc[:, 'question1'] = df_prep['question1'].apply(
                lambda _t: _stemming(_t))
            df_prep.loc[:, 'question2'] = df_prep['question2'].apply(
                lambda _t: _stemming(_t))

        return df_prep


class FeatureGenerator:
    def __init__(self,
                 exts: Tuple = ('cv', ),
                 aggs: Tuple = ('stack', ),
                 extra_features: Tuple[str] = 'all') -> None:
        assert len(exts) == len(aggs), \
            "Extractor and aggregator lists must be of the same length"
        self.extractors = [_get_extractor(ext) for ext in exts]
        self.extractor_names: Tuple[str] = exts
        self.aggregators = [_SUPPORTED_AGGREGATORS[agg] for agg in aggs]
        self.extra_features_creator = ExtraFeaturesCreator(extra_features)

    def set_params(self,
                   exts: Tuple = ('cv', ),
                   aggs: Tuple = ('stack', ),
                   extra_features: Tuple[str] = 'all') -> None:
        self.__class__(exts, aggs, extra_features)

    def fit(self, questions_df: pd.DataFrame, y: pd.DataFrame = None):
        self.extractors = [ext if name.startswith('spacy') else ext.fit(
            questions_df.values.flatten()) for name, ext in zip(
            self.extractor_names, self.extractors)]
        return self

    def transform(self, questions_df: pd.DataFrame, y: pd.DataFrame = None):
        agg_features = []
        for name, ext, agg in zip(self.extractor_names, self.extractors,
                                  self.aggregators):
            # we apply the extractor to each question
            if name.startswith('spacy'):
                print("Using spacy word embedding: please WAIT, "
                      "this may take some time")
                # then we use a spacy embedding
                x_q1 = questions_df.iloc[:, 0].apply(
                    lambda x: ext(x).vector)
                x_q2 = questions_df.iloc[:, 1].apply(
                    lambda x: ext(x).vector)

            else:
                x_q1 = ext.transform(questions_df.iloc[:, 0])
                x_q2 = ext.transform(questions_df.iloc[:, 1])

            # and we aggregate them
            x_agg = agg(x_q1, x_q2)
            agg_features.append(x_agg)

        if len(self.extra_features_creator.features_functions) != 0:
            # in parallel, we compute the extra features
            x_extra: np.ndarray = self.extra_features_creator.transform(
                questions_df)
            # finally, we merge them
            return hstack((hstack(agg_features), x_extra))
        return hstack(agg_features)


def _get_extractor(ext: str):
    if ext in ['spacy_small', 'spacy_medium']:
        _spacy_version: str = _SUPPORTED_EXTRACTORS[ext]
        try:
            import spacy
            spacy.load(_spacy_version)
        except OSError:
            os.system(f'python -m spacy download {_spacy_version}')
        finally:
            import spacy
            return spacy.load(_spacy_version)
    else:
        return _SUPPORTED_EXTRACTORS[ext]


class ExtraFeaturesCreator:
    def __init__(self, features_to_add: Union[Tuple, str]) -> None:
        if isinstance(features_to_add, str):
            assert features_to_add == 'all', "Unrecognized extra features list"
            self.features_functions: Dict[str, callable] = \
                _SUPPORTED_EXTRA_FEATURES
        else:
            self.features_functions = {
                _n: _SUPPORTED_EXTRA_FEATURES[_n] for _n in features_to_add}

    def transform(self, questions_df: pd.DataFrame) -> np.ndarray:
        if len(self.features_functions) == 0:
            raise ValueError("There is no extra features to be aggregated")

        for _c in ('question1', 'question2'):
            questions_df[_c] = questions_df[_c].str.split()
        extra_features = pd.DataFrame()

        for _f_name, _f_function in self.features_functions.items():
            extra_features[_f_name] = questions_df.apply(
                lambda x: _f_function(x.question1, x.question2), axis=1)

        return extra_features.values


# ##################################################################
# Grid Search functions
# ##################################################################

def get_param_grid(name: str, seed: int):
    """Returns param grid depending on model name.

    Args:
        name: model name.
        seed: random seed.
    """
    preprocessor_grid = {
        "preprocessor__remove_stop_words": [True, False],
        "preprocessor__remove_punctuation": [True, False],
        "preprocessor__to_lower": [True, False],
        "preprocessor__apply_stemming": [True, False],
        "preprocessor__british": [True, False],
    }
    exts = list(_SUPPORTED_AGGREGATORS.keys())
    aggs = list(_SUPPORTED_AGGREGATORS.keys())
    combinations_exts = [list(comb) for i in range(
        1, len(exts) + 1) for comb in itertools.combinations(exts, i)]
    combinations_aggs = [list(comb) for i in range(
        1, len(aggs) + 1) for comb in itertools.combinations(aggs, i)]

    generator_grid = {
        "generator__ext": combinations_exts,
        "generator__agg": combinations_aggs,
        "generator__extra_features": list(_SUPPORTED_EXTRA_FEATURES.keys())
    }

    classifier_grid = {
        "LogisticRegression": {
            "classifier__random_state": [seed],
            "classifier__penalty": ["l2"],
            "classifier__C": [0.01, 0.1, 1],  # , 10, 100]
        },
        "RandomForestClassifier": {
            "classifier__random_state": [seed],
            "classifier__n_estimators": [10, 50, 100],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__bootstrap": [True, False],
        },
        "SVC": {
            "classifier__random_state": [seed],
            "classifier__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "classifier__C": [0.01, 0.1, 1, 2],
            "classifier__gamma": ["scale", "auto"],
        },
        "KNeighborsClassifier": {
            "classifier__n_neighbors": [3, 5, 7],
            "classifier__weights": ["uniform", "distance"],
        },
        "GradientBoostingClassifier": {
            "classifier__random_state": [seed],
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [3, 5, None],
            "classifier__learning_rate": [0.01, 0.1, 1],
            "classifier__subsample": [0.5, 0.8, 1.0],
        },
        "AdaBoostClassifier": {
            "classifier__random_state": [seed],
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.01, 0.1, 1],
            "classifier__algorithm": ["SAMME", "SAMME.R"],
        },
        "XGBClassifier": {
            "classifier__random_state": [seed],
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [3, 5, 10],
            "classifier__learning_rate": [0.01, 0.1, 1],
            "classifier__subsample": [0.5, 0.8, 1.0],
        },
        "GaussianNB": {},
        "BernoulliNB": {},
        "LinearDiscriminantAnalysis": {
            "classifier__solver": ["svd", "lsqr", "eigen"],
            "classifier__shrinkage": [None, "auto", 0.1, 0.5, 0.9],
            "classifier__n_components": [None, 1],
            "classifier__store_covariance": [True, False],
            "classifier__tol": [1e-4, 1e-3, 1e-2],
        },
        "QuadraticDiscriminantAnalysis": {
            "classifier__reg_param": [0.0, 0.1, 0.5, 1.0],
            "classifier__store_covariance": [True, False],
            "classifier__tol": [1e-4, 1e-3, 1e-2],
        },
        "CatBoostClassifier": {
            "classifier__random_state": [seed],
            "classifier__iterations": [500, 1000],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__depth": [4, 6, 8],
            "classifier__l2_leaf_reg": [1, 3, 5],
            "classifier__border_count": [32, 64, 128],
            "classifier__loss_function": ["Logloss", "CrossEntropy"],
        },
    }
    return preprocessor_grid | generator_grid | classifier_grid[name]
