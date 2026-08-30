# Classificação de Recuperação Cardíaca com Random Forest

Projeto de aprendizado de máquina que utiliza o conjunto de dados Cleveland Heart Disease para classificar perfis de recuperação cardíaca em três categorias: `BOA`, `MODERADA` e `RUIM`.

O projeto implementa um pipeline completo em Python: leitura e limpeza dos dados, criação da variável-alvo, separação dos dados, treinamento de um modelo Random Forest, avaliação das previsões e geração de visualizações.

> **Aviso:** este é um projeto acadêmico de aprendizado de máquina. As classes de recuperação são criadas por regras definidas no próprio código e não representam um diagnóstico ou prognóstico médico validado.

## Objetivo

Investigar se informações de repouso e características dos pacientes permitem estimar uma classe de recuperação construída a partir de variáveis de exames de esforço.

Para reduzir o risco de vazamento de dados, as variáveis utilizadas diretamente na criação das classes (`thalach`, `exang`, `oldpeak` e `slope`) não são fornecidas ao modelo durante o treinamento.

## Como o projeto funciona

1. Carrega as 303 observações do arquivo `processed.cleveland.data`.
2. Identifica valores ausentes e remove as linhas incompletas, mantendo 297 observações.
3. Cria a variável `recuperacao` com base em uma pontuação calculada a partir de quatro variáveis relacionadas ao exame de esforço.
4. Separa os dados em 75% para treino e 25% para teste, preservando a proporção das classes.
5. Treina um `RandomForestClassifier` com 200 árvores e balanceamento automático das classes.
6. Avalia o modelo no conjunto de teste e por validação cruzada estratificada com 10 folds.
7. Exporta as principais análises para `resultados_rf.png`.
8. Executa quatro previsões individuais como verificação final do pipeline.

## Variáveis utilizadas no treinamento

O modelo é treinado com nove variáveis:

| Variável | Informação representada |
| --- | --- |
| `age` | Idade |
| `sex` | Sexo |
| `cp` | Tipo de dor no peito |
| `trestbps` | Pressão arterial em repouso |
| `chol` | Colesterol sérico |
| `fbs` | Indicador de glicemia em jejum acima de 120 mg/dl |
| `restecg` | Resultado do eletrocardiograma em repouso |
| `ca` | Número de vasos principais observados por fluoroscopia |
| `thal` | Resultado associado ao exame de perfusão cardíaca |

A descrição completa das variáveis está disponível na página oficial do [Heart Disease Dataset, da UCI](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease).

## Resultados obtidos

Com a configuração atual e `random_state=42`, o projeto apresentou:

| Métrica | Resultado |
| --- | ---: |
| Acurácia no conjunto de teste | 49,33% |
| Acurácia média na validação cruzada | 53,20% |
| Desvio-padrão da validação cruzada | 9,09% |

As cinco variáveis com maior importância calculada pelo modelo foram:

1. `age`: 0,2083
2. `chol`: 0,1988
3. `trestbps`: 0,1800
4. `cp`: 0,1209
5. `thal`: 0,1126

![Resultados do modelo Random Forest](resultados_rf.png)

O gráfico reúne a distribuição das classes, a importância das variáveis, a matriz de confusão e a acurácia obtida em cada fold da validação cruzada.

## Tecnologias utilizadas

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn

## Estrutura do projeto

```text
BCC325-main/
├── ClassificacaoRCE.py          # Pipeline de preparação, treino e avaliação
├── processed.cleveland.data     # Conjunto de dados utilizado
├── resultados_rf.png            # Visualizações geradas pelo script
└── README.md                     # Documentação do projeto
```

## Como executar

### Pré-requisito

Tenha o Python 3 instalado. Para verificar:

```bash
python3 --version
```

No Windows, caso `python3` não seja reconhecido, use `python` nos comandos abaixo.

### 1. Abra a pasta do projeto

```bash
cd BCC325-main
```

É importante executar o script dentro dessa pasta, pois ele procura o conjunto de dados por meio de um caminho relativo.

### 2. Crie e ative um ambiente virtual

Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instale as dependências

```bash
python3 -m pip install numpy pandas matplotlib scikit-learn
```

No Windows, se necessário:

```powershell
python -m pip install numpy pandas matplotlib scikit-learn
```

### 4. Execute o projeto

```bash
python3 ClassificacaoRCE.py
```

No Windows, se necessário:

```powershell
python ClassificacaoRCE.py
```

O terminal exibirá a distribuição das classes, as métricas, o relatório de classificação, as variáveis mais importantes e as previsões dos casos de teste. Ao final, o arquivo `resultados_rf.png` será criado ou atualizado na pasta do projeto.

## Decisões técnicas demonstradas

- Tratamento explícito de valores ausentes.
- Criação de uma variável-alvo por regras reproduzíveis.
- Prevenção de vazamento entre as variáveis usadas no alvo e as usadas no treinamento.
- Divisão estratificada entre treino e teste.
- Balanceamento de classes no Random Forest.
- Validação cruzada estratificada para avaliar a variação do desempenho.
- Uso de matriz de confusão, precision, recall, F1-score e importância de variáveis.
- Geração automatizada de visualizações para análise dos resultados.

## Limitações

- A variável-alvo não vem pronta no conjunto de dados: ela é criada por uma heurística implementada no código.
- Após a remoção de valores ausentes, o treinamento utiliza somente 297 observações.
- A acurácia média da validação cruzada foi de 53,20%, indicando espaço para investigação e melhoria.
- Os quatro casos individuais no final do script são verificações manuais do funcionamento e não constituem validação clínica.
- O projeto não deve ser utilizado para decisões médicas.

## Possíveis evoluções

- Comparar o Random Forest com outros algoritmos de classificação.
- Avaliar estratégias de imputação em vez de remover observações incompletas.
- Realizar ajuste sistemático de hiperparâmetros.
- Analisar métricas por classe com maior profundidade.
- Separar o pipeline em módulos e adicionar testes automatizados.
