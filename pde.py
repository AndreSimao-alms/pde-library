# Função -> aghata_efeito (fabi_efeito)

#Função adaptada do "fabi_efeito" utilizada no Octave pertencente ao Prof.Dr.Edenir Pereira Filho**\
#Canal do Youtube: https://www.youtube.com/c/EdenirPereiraFilho

#Funcao para calcular efeito de planejamento fatorial\
#X = matriz contendo os efeitos que serão calculados\
#y = vetor contendo a resposta\
#erro_efeito=erro de um efeito. Sera 0 se nao forem feitas replicas\
#t=valor de t correspondente ao número de graus de liberdade do erro de um efeito. Sera 0 se nao forem feitas replicas.
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns


class Aghata_efeito:

    def __init__(self, x, y, erro_efeito, t):
        self.X = x
        self.y = y
        self.erro_efeito = erro_efeito
        self.t = t
        self.inicio = [0]
        self.centro = []
        self.fim = []
        self.gauss = []

    @property
    def matrix_x(self):
        return self.X

    @property
    def vetor_y(self):
        return self.y

    @property
    def efeito(self):  # Retorna valores do produto entre efeitos e resposta
        return (self.X.T * self.y).T

    @property
    def __n_efeito(self):  # Retorna dimensões da matriz com os efeitos (valor_codificado*resposta)
        return self.X.shape

    @property
    def __indice_efeitos(self):  # Retorna lista com respectivas interações
        return self.X.T.index

    @property
    def __gerar_inicio_centro_fim_gauss(self):  # Retorna os valosres da gaussiana
        for i in range(self.__n_efeito[1]):
            self.fim.append(self.inicio[i] + (1 / self.__n_efeito[1]))
            self.inicio.append(self.fim[i])
            self.centro.append((self.inicio[i] + self.fim[i]) / 2)
            self.gauss.append(norm.ppf(self.centro))
        return self.gauss

    def calcular_efeitos(self):  # Retorna vetor com efeitos
        return (np.einsum('ij->j', self.efeito)) / (self.__n_efeito[0] / 2)  # np.einsum -> função que soma
        # colunas de uma matriz

    def calcular_porcentagem_efeitos(self):  # Retorna vetor com probabilidade
        return (self.calcular_efeitos() ** 2 / np.sum(self.calcular_efeitos() ** 2)) * 100

    def definir_gaussiana(self):  # Retorna os valosres da gaussiana
        return self.__gerar_inicio_centro_fim_gauss[self.__n_efeito[1] - 1]

    def porcentagem_efeitos(self):  # Retorna gráfico de barras de Porcentagem x Efeitos
        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 9))
        sns.load_dataset("tips")
        porcentagem_efeitos = sns.barplot(x='Efeitos', y='%', color='purple', data=pd.DataFrame(
            {'Efeitos': self.__indice_efeitos, '%': self.calcular_porcentagem_efeitos()}))
        porcentagem_efeitos.set_title('Gráfico de Probabilidades', fontsize=16, fontweight='black')
        return plt.show()

    def __etiqueta(self):  # Demarca os pontos no gráfico de probabilidades
        for i, label in enumerate(self.__sort_efeitos_probabilidades().index):
            plt.annotate(label, (self.__sort_efeitos_probabilidades()['Efeitos'].values[i],
                                 self.definir_gaussiana()[i]))

    def __sort_efeitos_probabilidades(self):  # Retorna dataframe ordenado de maneira crescente com valores de efeitos
        data = pd.DataFrame({'Efeitos': self.calcular_efeitos()}, index=self.__indice_efeitos)
        data = data.sort_values('Efeitos', ascending=True)
        return data

    def __definir_ic(self):  # Retorna conjunto de pontos do IC
        return np.full(len(self.definir_gaussiana()), self.erro_efeito * self.t)

    def grafico_probabilidades(self):  # Retorna gráfico de probabilidades
        plt.figure(figsize=(8, 9))
        plt.scatter(self.__sort_efeitos_probabilidades()['Efeitos'],
                    self.definir_gaussiana(), s=40, color='darkred')
        plt.title('Gráfico de Probabilidades', fontsize=18, fontweight='black', loc='left')
        plt.ylabel('z')
        plt.xlabel('Efeitos')
        plt.grid()
        self.__etiqueta()
        self.__verificar_ic()

    def __verificar_ic(self):
        if self.erro_efeito == 0 or self.t == 0:
            return plt.show()
        else:
            plt.plot(-self.__definir_ic(), self.definir_gaussiana(), color='darkred')
            plt.plot(0 * self.__definir_ic(), self.definir_gaussiana(), color='black')
            plt.plot(self.__definir_ic(), self.definir_gaussiana(), color='darkred')
            return plt.show()

    def aghata_efeito(self):
        return plt.show(self.grafico_probabilidades(), self.porcentagem_efeitos())
