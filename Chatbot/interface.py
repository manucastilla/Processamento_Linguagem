import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re
from random import randint
import requests
import json


def perguntar_usuario(sim):
    if sim == 1:
        print(" ")
        return input("O que você gostaria saber de Foxbot? ")

    else:
        return input("Você quer saber outra coisa?  ")


def limpa_tudo(df):
    # removendo minusculas
    df = df.lower()
    # remove todos os símbolos
    df = re.sub(r"\W", " ", df)
    # tirar o foxbot
    df = df.replace("foxbot", "")
    return df

# variar qual é o clima


def pegar_clima():
    cidade = input("Em qual cidade você está?")
    cidade = str(cidade)
    x = requests.get(
        'http://api.openweathermap.org/data/2.5/weather?q=Campinas&appid=1fe702bf6d43115c44bd4cb68f759de6')
    json_obj = x.json()
    temp_k = json_obj['main']['temp']
    temp = (temp_k - 273.15)
    weather = json_obj['weather'][0]["description"]

    print(
        f"A descrição do dia é {weather} e a temperatura do dia está em torno de {temp}°C!")


def pegar_ar(ar):
    if ar:
        print("ar ligado")
    else:
        print("ar desligado")


def pegar_luz(luz):
    if luz:
        print("Luz acessa")
    else:
        print("Luz apagada")


def pegar_conta(conta):
    print(f"Você possui R${conta} na sua conta !")


def fazer_correcao(model, intencoes, counts):
    print("O que você espera outro resulado... Qual era esse?")
    print("[1] Obter informações relativas ao clima")
    print("[2] Interagir com a luz ou o ar-condicionado")
    print("[3] Consultar saldo da conta")
    print(" ")

    while True:
        cor = input("Insere aqui o número que você deseja:  ")

        tipos = ["Obter informações relativas ao clima",
                 "Interagir com a luz ou o ar-condicionado",
                 "Consultar saldo da conta"]

        if cor in ["1", "2", "3"]:
            model.partial_fit(counts, [tipos[int(cor) - 1]], classes=intencoes)

            print("*"*30)
            print("Obrigada por este feedback :)")
            print("*"*30)
            break
        else:
            continue


def main():

    print("Olá, sou o Foxbot. que tipo de pergunta você deseja saber?")
    print("1. Saber do clima")
    print("2. Saber do ar ou luz")
    print("3. Saber do saldo da conta")
    print(" ")

    with open("model_geral.sav", "rb") as f:
        vectorizer, model = pickle.load(f)

    luz = randint(0, 2)
    ar = randint(0, 2)
    conta = randint(0, 7_000)

    intencoes = model.classes_
    # print(intencoes)

    while True:
        inputUsuario = perguntar_usuario(1)
        print(" ")
        inputUsuario = limpa_tudo(inputUsuario)

        if inputUsuario == "adeus":
            print("*"*30)
            print("Obrigado por vir falar comigo :)")
            print("*"*30)
            break

        counts = vectorizer.transform([inputUsuario])
        y_pred = model.predict(counts)

        if y_pred[0] == "Não sei":
            print(y_pred[0])
        else:
            print("", end="")

        if y_pred[0] == "Obter informações relativas ao clima":
            pegar_clima()

        elif y_pred[0] == "Interagir com a luz ou o ar-condicionado":

            while True:

                print("Você deseja Interagir com a luz ou com o ar-condicionado?")
                print("[1] Luz")
                print("[2] Ar-condicionado")
                a_l = input("Insere um dos números: ")
                a_l = limpa_tudo(a_l)

                if int(a_l) == 1:
                    pegar_luz(luz)
                    break
                elif int(a_l) == 2:
                    pegar_ar(ar)
                    break

        elif y_pred[0] == "Consultar saldo da conta":
            pegar_conta(conta)

        else:
            pass

        satisfeito = input("Usuário, você está satisfeito? [y/n]:  ")
        if satisfeito == "n":
            fazer_correcao(model, intencoes, counts)
            pass
        else:
            continue


if __name__ == "__main__":
    main()
