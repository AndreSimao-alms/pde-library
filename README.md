# pde-library
Biblioteca voltada para planejamento de experimentos inspirado nas rotinas de dados do octave criado pelo Prof.Dr.Edenir Pereira Filho.

# Bibliotecas 
Como nesta seção será utilizado a biblioteca personalizada *pde.py*, atente-se sobre as versões dos pacotes abaixo em seu computador. Caso necessário, utilize os comandos abaixo para instalar os pacotes: \
\
`pip intall nome_biblioteca == 0.0.0` \
\
```$ conda create -n nome_ambiente python=versão nome_biblioteca=versão anaconda```

Utilize o *magic method* `"__version__"` para verificar as versões dos pacotes.

**- - - PACOTES NECESSÁRIOS - - -**

- **Pandas** -> versão  = 1.4.1
- **Numpy** -> versão  = 1.22.3
- **Scipy** -> versão  = 1.7.3
- **Matplotlib** -> versão = 3.5.1
- **Tabulate** -> versão  = 0.8.9

## Erro de um Efeito e t-value

Definir os efeitos significativos do experimento é o principal objetivo da classe Fabi_efeito, para isso, precisa-se determinar o **intervalo de confiança dos efeitos**[4] através do produto dos valores de **Erro de um Efeito** e **valor-t**. A equação 1 descreve o **erro experimental** ou o **desvio padrão dos valores de resposta do experimento**, onde  ![formula](https://latex.codecogs.com/svg.image?{\color{Magenta}x_i}) é enésima resposta, ![formula](https://latex.codecogs.com/svg.image?{\color{Magenta}\bar{x}}) é a média aritmética das respostas e `n` é o número de experimentos envolvidos. Com isso, pode-se calcular o valor do erro de um efeito, indicado pela equação 2, onde o termo `k` é o número de variáveis envolvidas no planejamento fatorial. Dessa forma, o intervalo de confiança que é apresentado no gráfico de probabilidades pelas linhas verticais vermelhas é calculado pelo produto do valor de t-Student tabelado em relação ao grau de liberdade das réplicas do ponto central, indicado pela equação 3.[3]

![formula](https://latex.codecogs.com/svg.image?\color{Magenta}\text{Desvio&space;padrao&space;do&space;efeitos}=\text{Erro&space;experimental}=\sqrt{\frac{\sum{(x_i-\bar{x})^2}}{n-1}}\qquad\qquad\qquad\text{(Eq.&space;1)}\\\\\text{Erro&space;de&space;um&space;efeito}=\frac{2\times&space;\text{Erro&space;Experimental}}{\sqrt{n\times2^k}}\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\text{(Eq.&space;2)}\\\\\text{Intervalo&space;de&space;confianca&space;dos&space;efeitos}\;=\;\text{erro&space;de&space;um&space;efeito}\times\text{t-{&space;}value}\qquad\qquad\quad\qquad\;\text{(Eq.&space;3)})

## Classe auxiliar CP (*Central Points*) 
A classe CP tem o objetivo de auxiliar as outras classes gerando alguns valores, por exemplo o **erro de um efeito**. Geralmente, estes cálculos serão para inserir aos atributos no momento de instanciar uma classe da biblioteca, sendo assim não necessário recorrer ao Excel, tornando o processo mais hábil no processamento de dados. Veremos aqui ao **exemplo 1** dois métodos: primeiro, `pde.CP(y, k).erro_efeito()`, onde será necessário atribuir os valores de `y`, que são as respostas do ponto central e `k`, o número de variáveis envolvidas; segundo, a clássica `pde.CP.invt(df_a)`, que calcula o valor de t para distribuição bimodal t-Student para 95% de confiança, onde `df_a` é o grau de liberdade dos pontos centrais.

## Classe Regression2
Esta classe é responsável por gerar um modelo de regressão e realizar o seu ajuste através da *analisys of variance* (ANOVA), de modo que será calculado os valores dos coeficientes da equação do modelo e seus respectivos erros, os erros por sua vez são gerados após a verificação do *teste F*, onde a construção do intervalo de confiança será dado pela média quadrática dos resíduos, quando não há falta de ajuste do modelo, ou pela média quadrática de falta de ajuste, quando há falta de ajuste. Assim, uma vez gerado os resultados, o usuário terá que avaliar os coeficientes insignificantes para o modelo para, posteriormente, excluí-los.

### Parâmetros obrigatórios no método *regression2()*

#### Matriz X (*X*):
Valores codificados dos coeficientes do modelo.

#### Vetor y (*y*) :
Valores das respostas experimentais  

#### Soma Quadrática do Erro Puro (*SSPE*):
A soma quadrática é obtida através dos valores das réplicas do ponto central decrita pela equação 4. A classe *CP* também fornece um método para o cálculo do atributo. Utilize o comando `pde.CP(valores_centrais).SSPE()`. Para saber mais use `help(pde.CP.SSPE())`.

![formula](https://latex.codecogs.com/svg.image?{{\color{Magenta}SQ_{ep}=\frac{\sum{(y_{i}-\bar{y})^2}}{n-1}\;\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\text{(Eq.&space;4)}&space;})

#### Graus de Liberdade (*df*):
Graus de liberdade do ponto central, ou seja, ![formula](https://latex.codecogs.com/svg.image?{\color{Magenta}&space;N_{Replicas}-1&space;}).

###  Métodos auxiliares no método *regression2()*
O método `pde.Regression2().regression2()` possui alguns atributos no momento de instanciar a classe que são opcionais  de automação do tratamento de dados, segue as suas descrições:

#### self_turning:
O `self_turning`, que por padrão está configurado como `False`, este quando configurado como `True`, definirá se o modelo possui falta de ajuste automaticamente. O seu algoritmo é definido com um comando de seleção onde verifica a seguinte condição:

![formula](https://latex.codecogs.com/svg.image?{\color{Magenta}\text{Teste}\:F_{2}\:<\:F2_{tabelado}\;ou\;F1_{tabelado}\:<\:\text{Teste}\:F_1}\\\\\color{Magenta}\text{Teste}\:F_{1}=\frac{MS_{Reg}}{MS_{Res}}\quad\:e\quad\text{Teste}\:F_{2}=\frac{MS_{LoF}}{MS_{ep}}&space;)

Tendo em vista isso, atente-se se este comando se enquadra em dados em contextos que necessitam de uma análise mais cautelosa.  

#### auto (Não recomendável): 
O parâmetro `auto` também é do tipo booleano, que por padrão está configurado como `False`, este comando quando configurado como `True`, excluirá os coeficientes insignificantes e também os tornará nulo estes coeficientes na lista gerada através do método auxiliar `pde.Regression2.model_coefients()`. No entanto, o seu uso não é recomendável quando existe réplicas ao excluir colunas dos dados, pois ainda é inexistente a função que gera os valores médios das respostas das réplicas. Assim, no caso do exemplo 3 do Tutorial da Química Nova é possível utilizar este comando, pois se trata de um planejamento fatorial de Doerlert que não possui réplicas ao eliminar o coeficiente insignificante, não alterando valores das respostas e do grau de liberdade dos pontos centrais. 

### Métodos auxiliares da classe *Regression2*

A classe `Regression2` fornece métodos auxiliares que server para gerar alguns resultados a parte que é fornecido pelo o método mestre, de modo que são complementares para os atributos da classe *Super_fabi*. Segue suas descrições abaixo:

#### *model_coeficients()*:
O `model_coeficients` retorna uma lista com os valores dos coeficientes do modelo, este é uma ótima ferramenta quando queremos determinar a condição ótima experimental através dos gráficos de superfície e contorno do modelo fornecidos pela classe *Fabi_efeito*.

#### *show_ci()*:
Semelhante ao método anterior, este retorna os valores do intervalo de confiança dos coeficientes através de uma lista.

#### *dict_coefs_ci()*
Retorna uma lista contendo dicionários, sendo que as chaves são os coeficientes e os respectivos valores: `coef, coef + ci e coef - ci]`. 

#### *save_dataset()*
Cria um arquivo chamado *dataset.xlsx* no diretório contendo três páginas: a primeira, *ANOVA*, a tabela Anova gerada pelo método *regression2*; segundo, *coefs_ci*, os valores dos coeficientes do modelo e também os seus valores somados e subtraídos pelo intervalo de confiança; terceiro, *exp_pred*, os valores da resposta (vetor y) e os respectivos valores previstos pelo modelo.

## Classe *Super_fabi*
Responsável por retornar os gráficos de contorno e de superfície juntamente com a equação do modelo, valores codificados e reais para o valor máximo de sinal. Diferente da classe *Regression2* esta não possui parâmetros auxiliares, contendo no total 9 atributos obrigatórios, estes são: 

#### *coefs*:
Coeficientes do modelo de regressão que precisa ser *type list*, estes valores podem ser acessados com o método auxiliar *Regression2.model_coeficients()*.

#### realmax1 e realmin1
Valor real máximo e mínimo para a variável 1, respectivamente.

#### realmax2 e realmin2
Valor real máximo e mínimo para a variável 2, respectivamente.

#### codmax1 e codmin1
Valor codificados máximo e mínimo para a variável 1, respectivamente.

#### codmax2 e codmin2
Valor codificados máximo e mínimo para a variável 1, respectivamente.

### Principais métodos da classe *Super_fabi*

A classe *Super_fabi* não foi totalmente encapsulada com métodos privados para que p usuário tenha acesso aos valores gerados para a construção dos gráficos, assim há diversos métodos exposto que não serão apresentados nesta apostila. 

#### *superficie(matrix_X = None, vector_y = None, scatter=False)*:
Método mais importante da rotina, uma vez que este gera os resultados esperados pela classe, ou seja, a criação dos gráficos de superfície e de contorno juntamente com a equação do modelo e a condição experimental valores de resposta máxima. Este método possui um parâmetro auxiliar chamado de `scatter`, que está configurado por padrão por `False`, quando este recebe `True` e é informado um *dataframe* com os valores codificados dos coeficientes *b1* e *b2* através do parâmetro `matrix_X` e as respostas experimentais pelo parâmetro `vector_y` será construído os pontos experimentais do planejamento fatorial no gráfico de contorno. 

#### *z(meshgrid=None, x=None,  y=None, manual=False)*:
Este retorna os valores previstos pelo modelo de três maneiras: primeiro, através de um vetor com 100 itens quando `meshgrid=False`; segundo, através de uma matriz com 100 colunas e 100 linhas quando `meshgrid=True`; terceiro, um único valor que é calculado manualmente pelo modelo, para isso, mantenha o parâmetro *meshgrid* em `None` e configure `manual=True` e também os valores codificados de x e y que será calculado pela equação do modelo. 

### *Property's* da classe *Super_fabi* 
Embora o assunto programação orientada a objetos não foi tratada nesta playlist, este tópico será abordados brevemente nesta seção. A *property's* de uma classe são atributos ou atributos modificados que são facilmente acessados, por exemplo na biblioteca Pandas quando queremos acessar as dimensões de um dataframe e usamos o comando shape da seguinte maneira, `pandas.dataframe.shape`. Note que não precisamos colocar os parênteses que comumente estão presentes em métodos. Tendo em vista isso, segue abaixo os valores que podem ser acessados pela classe:

#### *maxcod* 
Retorna valores das coordenadas do sinal máximo para as variáveis codificadas.

#### *maxreal*
Retorna valores das coordenadas do sinal máximo para as variáveis reais.

#### *zmax*
Retorna o valor do sinal máximo do modelo.

### Sobre o método auxiliar *solver diff 

Para obter as condições ideais de experimentação através das derivadas, será utilizado um método auxiliar da classe *Super_fabi* chamado *solver_diff()*. Os coeficiente serão determinados a partir do sistema de equações formado com a derivação parcial da equação do modelo, dessa maneira, os coeficientes que descrevem as condições ideais para experimentação será através das raízes encontradas. Por fim, o valor da resposta máxima será o resultado previsto pelo modelo utilizando os coeficientes encontrados no cálculo. Outro importante ser mencionado é o método de resolução do sistema de equações, que no caso do método *solver_diff()* será por meio das propriedades de matrizes, a biblioteca disponibiliza para o planejamento fatoriais envolvendo 2,3 e 4 variáveis.  

# Material Suplementar
- **Playlist 8: Tutorial Química Nova - Introdução**
\
https://www.youtube.com/watch?v=ai6mb6KENmw&list=PL4CuftF4l_fBDpo7b57Hn95sGEtkyChvA 

- **Playlist 9: Tutorial Química Nova - Exemplo 1**
\
https://www.youtube.com/watch?v=mVFS_wdtj6I&list=PL4CuftF4l_fDA0BXsxRGZoE9s0j7yPZ30 

- **Playlist 10: Tutorial Química Nova - Exemplo 2**
\
https://www.youtube.com/watch?v=iqTaAYSS0Fk&list=PL4CuftF4l_fCFGmzWcfpdY33r0Focz8jI 

- **Playlist 11: Tutorial Química Nova - Exemplo 3**
\
https://www.youtube.com/watch?v=y6Vdm0WBRiU&list=PL4CuftF4l_fAbKkSS1i-eBGFcPhgMfIyJ 

- **Playlist 12: Tutorial Química Nova - Exemplo 4**
\
https://www.youtube.com/watch?v=uYdnfxo54QQ&list=PL4CuftF4l_fA5DLOY9PLdFZdWIl1dFgLY 


# Referências 
1. Pereira, Fabíola Manhas Verbi, and Edenir Rodrigues Pereira-Filho. "Aplicação de programa computacional livre em planejamento de experimentos: um tutorial." Química Nova 41 (2018): 1061-1071.


2. Santos, G. S.; Silva, L. O. B., Santos Júnior, A. F.; Silva, E. G. P.,Santos, W. N. L.; J. Braz. Chem. Soc. 2018, 29, 185.


3. Pereira Filho, E. R.; Planejamento fatorial em química: maximizando a obtenção de resultados, Edufscar: São Carlos, 2015.


4. Barros Neto, B.; Scarminio, I. S.; Bruns, R. E.; Como fazer experimentos, Bookman: Porto Alegre, 2010.


5. Ferreira, S. L. C.; Santos, W. N. L.; Quintella, C. M.; Barros Neto, B.; Bosque-Sendra, J. M.; Talanta 2004, 63, 1061.
