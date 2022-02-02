# pde-library
  Biblioteca voltada para planejamento de experimentos inspirado nas rotinas de dados do octave criado pelo Prof. Dr. Edenir Pereira Filho.

### Objetivo geral: 
  Adaptar rotinas realizadas no octave na linguagem python, possibilitando novas funções através das funcionalidades da linguagem

### Qual o contexto do projeto? 
Atualmente estou cursando o curso de bacharelado em química pela UFSCar e este projeto faz parte da proposta do meu trabalho de conclusão de curso. Juntamente com o desenvolvimento desta biblioteca, estou construindo um curso de python básico com foco em química, dessa maneira será contextualizado o uso da linguagem python em resolução de problemas reais na área de química. 

## Estrutura da biblioteca
A biblioteca dividirá, inicialmente, as rotinas "fabi_efeito", "super_fabi" e "regression2" do octave em classes, estas terão os métodos para a realização do tratamento dos dados. 

### Aghata_efeito (fabi_efeito)
Classe responsável para calcular efeito de planejamento fatorial\
Parâmetros:\
\
X = matriz contendo os efeitos que serão calculados\
y = vetor contendo a resposta\
erro_efeito = erro de um efeito. Sera 0 se nao forem feitas replicas\
t = valor de t correspondente ao número de graus de liberdade do erro de um efeito. Sera 0 se nao forem feitas replicas.\
\
#### Principais métodos da classe: Aghata_efeito
 - ##### aghata_efeito() -> Retorna gráfico de probabilidades com intervalo de confiança (ou não) e gráfico de porcentagem de efeitos x efeitos
 - ##### calcular_efeitos() -> Retorna vetor com efeitos
 - ##### calcular_porcentagem_efeitos() -> Retorna vetor com probabilidade
 - ##### definir_gaussiana() -> Retorna os valores da gaussiana
 - ##### __indice_efeitos() ->  Retorna lista com respectivos efeitos
 - ##### porcentagem_efeitos() -> Retorna gráfico de barras de Porcentagem x Efeitos
 - ##### grafico_probabilidades() -> Retorna gráfico de probabilidades
    
