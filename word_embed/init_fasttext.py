from gensim.models.fasttext import FastText as FT
from gensim.utils import tokenize
from gensim import utils
import glob
import sys
import yaml


def iter_doc(path):

    with utils.open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            yield list(tokenize(line))


def init_model(path):

    model = FT(size=150, window=5, min_count=3, workers=4)
    model.build_vocab(sentences=iter_doc(path))
    model.train(corpus_file=path,
                epochs=model.epochs,
                total_examples=model.corpus_count,
                total_words=model.corpus_total_words,
                model='skipgram')

    show_vocab_size(model)

    return model


def update_model(model, path):

    print(path)

    model.build_vocab(sentences=iter_doc(path), update=True)
    model.train(corpus_file=path,
                epochs=model.epochs,
                total_examples=model.corpus_count,
                total_words=model.corpus_total_words,
                model='skipgram')

    show_vocab_size(model)

    return model


def repeat_update(model, file_list):

    for i, fname in enumerate(file_list):
        print('{}/{}'.format(i, len(file_list)))
        model = update_model(model, fname)

    model.save('../data/word_vector/training_model/KT_model')


def show_vocab_size(model):
    print(len(model.wv.vocab))


def main():

    with open(sys.argv[1], encoding='utf-8-sig') as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    print(config)
    path = config['path']
    init = config['init']
    pretrained_model = config['pretrained_model']

    file_list = [fname for fname in glob.glob(path)]

    if init:

        model = init_model(file_list[0])
        file_list.pop(0)
        repeat_update(model, file_list)

    else:

        model = FT.load(pretrained_model)
        repeat_update(model, file_list)


if __name__ == '__main__':
    main()
