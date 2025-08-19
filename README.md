Telecom X - Análise Preditiva de Churn de Clientes (Parte 2)

Este projeto foca na análise e modelagem preditiva da evasão de clientes (churn) para a empresa fictícia Telecom X. O objetivo principal é utilizar dados históricos para construir um modelo de machine learning capaz de prever quais clientes têm maior probabilidade de cancelar seus serviços, permitindo que a empresa tome ações proativas de retenção.

🎯 Propósito da Análise
O objetivo central desta análise é desenvolver um modelo preditivo para identificar clientes com alto risco de churn. Com base nas variáveis mais influentes, a análise também visa gerar insights estratégicos que possam orientar a criação de campanhas de marketing, ofertas personalizadas e melhorias nos serviços para aumentar a fidelidade e a retenção de clientes.

📂 Estrutura do Projeto
O repositório está organizado da seguinte forma:

├── TelecomX_parte2_BR.ipynb   # Notebook principal com toda a análise, pré-processamento e modelagem.
├── dados_tratados.csv         # Conjunto de dados limpo e pré-tratado, pronto para ser carregado no notebook.
├── README.md                  # Este arquivo com a documentação do projeto.
└── /visualizacoes/            # (Opcional) Pasta para salvar os gráficos gerados durante a análise.
📊 Análise Exploratória de Dados (EDA) e Insights
Antes da modelagem, uma análise exploratória foi realizada para entender o comportamento dos dados e extrair insights iniciais.

Proporção de Churn
A análise inicial mostrou um desbalanceamento de classes, com aproximadamente 26.5% dos clientes tendo cancelado o serviço. Isso justifica o uso de técnicas como o SMOTE para balancear o conjunto de treino.

Tempo de Contrato vs. Evasão
Clientes com menor tempo de permanência são significativamente mais propensos a cancelar. O gráfico de boxplot abaixo ilustra que a mediana de meses de permanência para clientes que evadiram é muito inferior à dos que permaneceram.

(Exemplo de gráfico que pode ser gerado e salvo pelo notebook)

Matriz de Correlação
A matriz de correlação das variáveis numéricas (após a codificação) destacou que a evasão (Cancelou_Yes) tem uma forte correlação negativa com Meses_Permanencia e Tipo_Contrato_Two year, e uma correlação positiva com Tipo_Internet_Fiber optic e Metodo_Pagamento_Electronic check.

(Exemplo de gráfico que pode ser gerado e salvo pelo notebook)

⚙️ Preparação dos Dados
O processo de preparação dos dados foi crucial para garantir a qualidade dos modelos e consistiu nas seguintes etapas:

Codificação de Variáveis Categóricas: As variáveis categóricas (como Tipo_Contrato, Metodo_Pagamento, etc.) foram transformadas em formato numérico utilizando a técnica de one-hot encoding (pd.get_dummies). O parâmetro drop_first=True foi usado para evitar multicolinearidade.

Separação em Conjuntos de Treino e Teste: Os dados foram divididos em 70% para treino e 30% para teste (train_test_split). Foi utilizada a estratificação (stratify=y) para garantir que a proporção de clientes que cancelaram e não cancelaram fosse a mesma em ambos os conjuntos, o que é fundamental em casos de dados desbalanceados.

Balanceamento de Classes (SMOTE): Para corrigir o desbalanceamento de classes no conjunto de treino, aplicamos a técnica SMOTE (Synthetic Minority Over-sampling Technique). O SMOTE cria novas amostras sintéticas da classe minoritária (clientes que cancelaram), resultando em um conjunto de treino balanceado e melhorando a capacidade do modelo de aprender os padrões de churn.

Normalização de Dados: Para o modelo KNN, que é baseado em distância, as variáveis numéricas foram normalizadas com StandardScaler. Isso evita que variáveis com escalas maiores dominem o cálculo das distâncias. Esta etapa foi aplicada após o SMOTE e apenas para o modelo KNN.

🤖 Modelagem e Justificativa
Foram desenvolvidos e avaliados dois modelos distintos:

Random Forest Classifier: Um modelo baseado em árvores de decisão que não exige normalização dos dados. É robusto, lida bem com interações complexas entre variáveis e permite a análise de importância das features.

K-Nearest Neighbors (KNN): Um modelo baseado em distância que, por sua natureza, exigiu a normalização dos dados de treino e teste para funcionar corretamente.

Justificativa da Escolha do Modelo Final
Para um problema de previsão de churn, o custo de um Falso Negativo (prever que um cliente não vai cancelar, quando na verdade ele vai) é muito alto, pois representa uma oportunidade de retenção perdida. Portanto, a métrica mais importante para este caso é o Recall para a classe "Churn" (valor 1).

O Random Forest Classifier foi o modelo escolhido, pois apresentou um desempenho superior, com um recall de 0.65 para a classe de evasão, contra 0.61 do KNN, indicando que ele é mais eficaz em identificar os clientes que realmente irão cancelar.

Relatório de Classificação (Random Forest no conjunto de teste):

              precision    recall  f1-score   support

           0       0.86      0.81      0.83      1552
           1       0.55      0.65      0.60       561

    accuracy                           0.77      2113
   macro avg       0.71      0.73      0.72      2113
weighted avg       0.78      0.77      0.77      2113
Além disso, foi utilizado o GridSearchCV para otimizar os hiperparâmetros do Random Forest, garantindo a melhor performance possível.

🚀 Como Executar o Projeto
Siga os passos abaixo para executar a análise em seu ambiente local ou no Google Colab.

1. Pré-requisitos
Python 3.x

As bibliotecas listadas abaixo.

2. Instalação das Bibliotecas
Abra um terminal e execute o comando a seguir para instalar as dependências:

Bash

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
3. Execução do Notebook
Clone este repositório para a sua máquina local.

Certifique-se de que o arquivo dados_tratados.csv está no mesmo diretório que o notebook TelecomX_parte2_BR.ipynb.

Abra o notebook em um ambiente como Jupyter Notebook, JupyterLab ou Google Colab.

Execute as células do notebook em ordem sequencial para replicar a análise.

💡 Conclusão e Estratégias Propostas
A análise e o modelo preditivo forneceram insights valiosos sobre os fatores que levam à evasão de clientes. As variáveis mais importantes para prever o churn foram, em ordem: tempo de permanência, tipo de contrato, cobrança total e mensal, e o tipo de serviço de internet.

Com base nesses resultados, as seguintes estratégias são recomendadas para a Telecom X:

Foco nos Recém-Chegados: Criar um programa de onboarding para clientes novos, oferecendo suporte e benefícios nos primeiros meses.

Incentivar Contratos de Longo Prazo: Oferecer descontos ou vantagens para clientes que migrarem de contratos mensais para anuais ou bianuais.

Promover Serviços de Valor Agregado: Criar campanhas para incentivar a adesão a serviços como Suporte Técnico e Segurança Online, que demonstraram aumentar a retenção.

Ação Proativa com o Modelo Preditivo: Utilizar o modelo Random Forest para identificar mensalmente clientes com alta probabilidade de churn e direcioná-los para uma equipe de retenção com ofertas personalizadas.
