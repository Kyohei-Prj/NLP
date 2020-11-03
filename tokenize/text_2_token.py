from multiprocessing import Pool
import neologdn as neo
import MeCab
import glob
import re


def replacer(text, pattern, replace):

    return [re.sub(pattern, replace, sentence) for sentence in text]


def tokenizer(text_str):

    tag = '-Owakati'
    dic = ' -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd'
    tagger = MeCab.Tagger(tag + dic)

    tokens = tagger.parse(text_str)

    return tokens


def remove_stopwords(token_list):

    with open('../data/stopwords/jp_stopwords.txt', mode='r') as f:
        stopwords = f.read()

    stopwords = set(stopwords.split('\n'))

    processed_text = [token for token in token_list if token not in stopwords]

    return processed_text


def start_preprocess(path):

    with open(path, mode='r', encoding='utf-8') as f:
        text = f.readlines()

    # preprocess text
    text = [neo.normalize(sentence, repeat=2).lower() for sentence in text]
    text = replacer(text, r'<.+>', '')
    text = replacer(text, r'\d+', '')
    text = [sentence.rstrip() for sentence in text]
    pattern = r'[、。「」〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％：〔〕“”!"#$%&()*+,-./:;<=>?@^_`{|}~]'
    text = replacer(text, pattern, '')

    # tokenize text
    text_str = ' '.join(text)
    tokens = tokenizer(text_str)
    token_list = tokens.split(' ')

    # remove stopwords
    processed_text = remove_stopwords(token_list)
    processed_text = ' '.join(processed_text)

    # save to txt
    save_path = path.replace('text', 'processed_text')
    with open(save_path, mode='w') as f:
        f.write(processed_text)


def main():

    folder_list = [folder for folder in glob.glob('../data/wikipedia/text/*/')]

    for i, folder in enumerate(folder_list):
        print('{}/{}'.format(i, len(folder_list)))
        file_list = [fname for fname in glob.glob(folder + '*')]
        p = Pool(8)
        p.map(start_preprocess, file_list)
        p.close()
        p.terminate()


if __name__ == '__main__':
    main()
