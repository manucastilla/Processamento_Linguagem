import pandas as pd
from nltk import word_tokenize
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle


def limpa_tudo(df):
    # removendo minusculas
    df = df.lower()
    # remove todos os símbolos exceto $
    df = re.sub(r"\W", " ", df)
    # tirar o foxbot
    df = df.replace("foxbot", "")
    return df


def model_train(df):

    # limpando o dataset
    df["Sentença"] = df["Sentença"].apply(limpa_tudo)

    # separar em dataset de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        df["Sentença"], df["Intenção"], test_size=0.2, random_state=42)

    # classificador naive-Bayes
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X, y_train)

    countsTest = vectorizer.transform(X_test)

    # ou sera que um cross val é melhor?
    y_pred = model.predict(countsTest)
    # acc = accuracy_score(y_test, y_pred)
    return model, vectorizer


def main():

    # lendo o excel
    df = pd.read_excel("sentencas.xlsx")

    model, vectorizer = model_train(df)

    with open("model_geral.sav", "wb") as f:
        pickle.dump((vectorizer, model), f)


if __name__ == "__main__":
    main()
