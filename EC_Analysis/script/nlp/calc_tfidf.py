from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter
# from janome.tokenfilter import POSStopFilter
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import sys


def tokenize(df):

    tokenizer = Tokenizer()
    # token_filters = [POSStopFilter(['記号', '助詞', '助動詞'])]
    token_filters = [POSKeepFilter(['名詞'])]
    analysis = Analyzer(tokenizer=tokenizer, token_filters=token_filters)

    corpus = []
    for col in df.columns.to_list():
        df_tmp = df[col].dropna()
        sentence_concat = concat_str(df_tmp.values)
        token = analysis.analyze(sentence_concat)
        word_list = [word.surface for word in token]
        word_concat = concat_str(word_list)
        corpus.append(word_concat)

    return corpus


def concat_str(string_list):

    string_concat = ''
    for string in string_list:
        string_concat = string_concat + string + ' '

    return string_concat


def calc_tfidf(corpus):

    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(corpus).toarray()
    features = vec.get_feature_names()

    return features, tfidf


def sort_result(features, tfidf):

    result_list = []
    for values in tfidf:
        result_dict = dict(zip(features, values))
        result_dict_sort = {
            ky: val
            for ky, val in sorted(result_dict.items(),
                                  key=lambda item: item[1], reverse=True)
        }
        result_list.append(result_dict_sort)

    return result_list


def print_top_n(result_list, n):

    for result_dict in result_list:
        print('corpus')
        keyword_list = []
        tfidf_values_list = []
        for i, key in enumerate(result_dict.keys()):
            keyword_list.append(key)
            tfidf_values_list(result_dict[key])
            if i > n:
                break
        print()


def save_result(result_list, col_names, n):

    for result_dict, col in zip(result_list, col_names):
        word_list = []
        value_list = []
        print(col)
        for index, key in enumerate(result_dict.keys()):
            word_list.append(key)
            value_list.append(result_dict[key])
            if index > n:
                break
        df = pd.DataFrame({col: word_list, 'tfidf': value_list})
        save_path = '../../data/tfidf/' + 'tfidf_' + col + '.csv'
        df.to_csv(save_path, index=False, encoding='utf_8_sig')


def main():

    df = pd.read_csv(sys.argv[1])
    df.reset_index(inplace=True, drop=True)

    corpus = tokenize(df)
    features, tfidf = calc_tfidf(corpus)
    result_list = sort_result(features, tfidf)
    print(df.columns.to_list())
    save_result(result_list, df.columns.to_list(), int(sys.argv[2]))


if __name__ == '__main__':
    main()
