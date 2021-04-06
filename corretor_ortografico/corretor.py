from tqdm import tqdm
import json
import re
import nltk
from nltk import word_tokenize
from collections import Counter
import sys

vocab = {}
with open('vocab.jsonln') as json_file:
    vocab = json.load(json_file)

LOWERCASE = [chr(x) for x in range(ord('a'), ord('z') + 1)]
# https://www.ascii-codes.com/cp860.html
LOWERCASE_OTHERS = ['ç', 'á', 'â', 'ã', 'à', 'é',
                    'í', 'ó', 'ú', 'ê', 'î', 'ô', 'û', 'õ']  # etc.
LETTERS = LOWERCASE + LOWERCASE_OTHERS

# Leventhein com apenas uma unidade de distância de edição


def edit1(text):
    words = []

    # Fase 1: as remoçoes.
    for p in range(len(text)):
        new_word = text[:p] + text[p + 1:]
        if len(new_word) > 0:
            words.append(new_word)

    # Fase 2: as adições.
    for p in range(len(text) + 1):
        for c in LETTERS:
            new_word = text[:p] + c + text[p:]
            words.append(new_word)

    # Fase 3: as substituições.
    for p in range(len(text)):
        orig_c = text[p]
        for c in LETTERS:
            if orig_c != c:
                new_word = text[:p] + c + text[p + 1:]
                words.append(new_word)

    return set(words)

# Leventhein com duas unidades de distância de edição


def edit2(text):
    words1 = edit1(text)
    words2 = set()
    for w in words1:
        candidate_words2 = edit1(w)
        candidate_words2 -= words1
        words2.update(candidate_words2)
    words2 -= set([text])
    return words2

# encontrar as possíveis palavras corretas


def candidates(word):
    if word in vocab:
        candidatos = [word]
    else:
        candidatos = []
        candidatos += \
            [w for w in edit1(word) if w in vocab] \
            + [w for w in edit2(word) if w in vocab] \
            + [word]
    return candidatos


def probabilidade(word, n=sum(vocab.values())):
    if word in vocab:
        return vocab[word] / n
    else:
        return 0


def corretor(word):
    return max(candidates(word), key=probabilidade)


def corrigeFrase(frase):
    print(f"A frase a ser corrigida é: {frase}")
    print("------------------------------------")
    tokens = frase.split()
    fraseCorrigida = ""
    for palavra in tokens:
        if palavra in nltk.corpus.stopwords.words('portuguese'):
            fraseCorrigida += palavra + " "
        else:
            fraseCorrigida += corretor(palavra) + " "

    print("a frase corrigida é: ")
    return fraseCorrigida


def main():
    var = sys.argv[1]
    return corrigeFrase(var)


if __name__ == "__main__":
    print(main())
