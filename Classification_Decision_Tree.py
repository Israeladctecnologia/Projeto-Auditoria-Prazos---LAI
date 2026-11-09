# ==============================================================================
# PROJETO: ANÁLISE E AUDITORIA DE PRAZOS (Pedidos da LAI - Fala.BR)
# MODELO: ÁRVORE DE DECISÃO (CLASSIFICAÇÃO) - METODOLOGIA CRISP-DM
#
# Mapa do script em relação às fases do CRISP-DM:
#   Seções 2 a 4  -> Entendimento e Preparação dos Dados
#   Seções 5 a 6  -> Modelagem
#   Seções 7 a 10 -> Avaliação e interpretação
#   Seção 11      -> Uso do modelo em um pedido novo + limitações
# ==============================================================================

# ==============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS
# ==============================================================================

# Path ajuda a localizar o arquivo CSV na mesma pasta deste script, não importa de onde você execute.
from pathlib import Path

# Importa o Pandas para criação, manipulação e estruturação de dados em tabelas (DataFrames).
import pandas as pd

# Importa a função para dividir os dados em conjuntos distintos de treinamento e teste.
from sklearn.model_selection import train_test_split

# Importa a classe do algoritmo de Árvore de Decisão para classificação, a função para extrair suas regras em texto
# e a função para desenhar a árvore.
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# Importa funções para calcular métricas de desempenho: acurácia, precisão, recall, F1 e matriz de confusão.
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

# Importa o módulo PyPlot do Matplotlib para geração e customização de gráficos e figuras.
import matplotlib.pyplot as plt

# Importa a classe do scikit-learn para formatar e desenhar graficamente a matriz de confusão.
from sklearn.metrics import ConfusionMatrixDisplay


# ==============================================================================
# 2. CARREGAMENTO DOS DADOS E CONVERSÃO DAS DATAS
# ==============================================================================
# MUDANÇA EM RELAÇÃO AO CÓDIGO ORIGINAL: a linha "import DadosAbertos_LAI as dblai" tentava importar o CSV
# como se fosse um módulo Python, o que não funciona. Aqui o arquivo é lido com o pd.read_csv.

# Monta o caminho do arquivo CSV (que está na mesma pasta deste script).
caminho_csv = Path(__file__).parent / "DadosAbertos_LAI.csv"

# Lê o CSV. Dois detalhes do arquivo:
# - sep=';'  -> as colunas são separadas por ponto e vírgula (padrão do Excel em português);
# - encoding='cp1252' -> o arquivo foi salvo pelo Excel no padrão do Windows. Se usássemos UTF-8,
#   os acentos dariam erro ou apareceriam quebrados.
df = pd.read_csv(caminho_csv, sep=';', encoding='cp1252')

# O arquivo termina com uma linha totalmente vazia. Removemos qualquer linha sem IdPedido.
df = df.dropna(subset=['IdPedido']).reset_index(drop=True)

# Converte as colunas de data de texto ("02/01/2024") para o tipo data do Pandas.
# - format='%d/%m/%Y' diz que o formato é dia/mês/ano;
# - errors='coerce' transforma datas inválidas em vazio (NaT) em vez de parar o programa.
for coluna in ['DataRegistro', 'PrazoAtendimento', 'DataResposta']:
    df[coluna] = pd.to_datetime(df[coluna], format='%d/%m/%Y', errors='coerce')

# Mostra um resumo rápido para conferirmos se a leitura deu certo.
print("=== Dados carregados ===")
print(f"Registros: {len(df)} | Colunas: {df.shape[1]}")
print(f"Datas inválidas: {int(df[['DataRegistro', 'PrazoAtendimento', 'DataResposta']].isna().sum().sum())}\n")


# ==============================================================================
# 3. CRIAÇÃO DA VARIÁVEL-ALVO (TARGET)
# ==============================================================================

# Cria a coluna DentroDoPrazo: 1 se a resposta saiu até o prazo (DataResposta <= PrazoAtendimento), 0 se saiu depois.
# A comparação gera True/False e o astype(int) converte para 1/0.
df['DentroDoPrazo'] = (df['DataResposta'] <= df['PrazoAtendimento']).astype(int)

# Mostra quantos pedidos há em cada classe. ATENÇÃO: aqui já dá para ver que a base é muito desbalanceada.
print("=== Distribuição da variável-alvo ===")
print(df['DentroDoPrazo'].value_counts().rename({1: 'Dentro do prazo (1)', 0: 'Fora do prazo (0)'}))
print()


# ==============================================================================
# 4. SELEÇÃO E TRANSFORMAÇÃO DAS VARIÁVEIS DE ENTRADA (FEATURES)
# ==============================================================================

# Cria uma variável numérica nova: quantos dias de prazo o órgão tinha (PrazoAtendimento - DataRegistro).
# É uma informação que existe quando o pedido chega, e prazos maiores podem indicar pedidos mais complexos.
df['DiasDePrazo'] = (df['PrazoAtendimento'] - df['DataRegistro']).dt.days

# Escolha das variáveis de entrada e o motivo de cada exclusão:
# - FICAM: OrgaoDestinatario, AssuntoPedido, FoiProrrogado, FoiReencaminhado, OrigemSolicitacao, DiasDePrazo.
# - FORA (identificadores): IdPedido e IdSolicitante. São apenas "números de série"; não explicam atraso
#   e a árvore poderia decorá-los (overfitting) sem aprender nada útil.
# - FORA (vazamento de informação / data leakage): DataResposta, Decisao e EspecificacaoDecisao só existem
#   DEPOIS que o pedido foi respondido. DataResposta, além disso, é usada para criar o alvo, ou seja, daria a
#   "resposta da prova" ao modelo. Um modelo de apoio à auditoria precisa funcionar com o que se sabe antes.
# - FORA (val+
#  ajudam a distinguir um pedido do outro.
# - FORA (redundantes): DataRegistro e PrazoAtendimento já estão resumidas em DiasDePrazo.
colunas_entrada = ['OrgaoDestinatario', 'AssuntoPedido', 'FoiProrrogado',
                   'FoiReencaminhado', 'OrigemSolicitacao', 'DiasDePrazo']
X = df[colunas_entrada].copy()

# Seleciona a coluna alvo (1 = dentro do prazo, 0 = fora do prazo) e armazena no vetor y.
y = df['DentroDoPrazo']

# Existem 129 órgãos e 70 assuntos diferentes em apenas ~365 pedidos: muitas categorias aparecem 1 ou 2 vezes.
# Para não criar centenas de colunas, agrupamos as categorias raras (menos de 10 pedidos) em "Outros".
for coluna in ['OrgaoDestinatario', 'AssuntoPedido']:
    frequencia = X[coluna].value_counts()
    raras = frequencia[frequencia < 10].index
    X[coluna] = X[coluna].where(~X[coluna].isin(raras), 'Outros')

# Transformação das categorias em números pelo método ONE-HOT ENCODING (pd.get_dummies):
# cada categoria vira uma coluna própria com valor 0 ou 1. Exemplo: a coluna FoiProrrogado
# vira "FoiProrrogado_Sim" (1 = sim, 0 = não). Escolhemos esse método em vez de numerar as categorias
# (1, 2, 3...) porque numerar criaria uma falsa ordem entre elas (como se "Sim" fosse maior que "Não").
# - drop_first=True remove uma coluna redundante por variável (se não é "Sim", é "Não").
X = pd.get_dummies(X, drop_first=True, dtype=int)

print("=== Variáveis de entrada após a transformação ===")
print(f"{X.shape[1]} colunas numéricas: {list(X.columns)}\n")


# ==============================================================================
# 5. DIVISÃO DOS DADOS EM TREINO E TESTE
# ==============================================================================

# Divide X e y em conjuntos de treino (80%) e teste (20%):
# - test_size=0.20: Reserva 20% das amostras para avaliação do modelo.
# - random_state=42: Garante a mesma divisão dos dados em execuções futuras.
# - stratify=y: Preserva a mesma proporção de pedidos dentro/fora do prazo tanto no treino quanto no teste.
#   Isso é essencial aqui, porque os pedidos atrasados são raros; sem isso, o teste poderia não ter nenhum.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Treino: {len(X_train)} pedidos ({(y_train == 0).sum()} fora do prazo)")
print(f"Teste:  {len(X_test)} pedidos ({(y_test == 0).sum()} fora do prazo)\n")


# ==============================================================================
# 6. INSTANCIAÇÃO E TREINAMENTO DO MODELO
# ==============================================================================

# Cria o modelo de Árvore de Decisão:
# - max_depth=3: limita a profundidade a 3 níveis para evitar sobreajuste (overfitting) e manter as regras legíveis.
# - class_weight='balanced': como há pouquíssimos pedidos fora do prazo, esse ajuste dá mais peso a eles no treino,
#   para a árvore não aprender apenas "todo mundo responde no prazo".
# - random_state=42: resultado reproduzível.
modelo = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)

# Treina o algoritmo aprendendo as regras de divisão nos dados de treino (X_train e y_train).
modelo.fit(X_train, y_train)

# Nomes das classes na ordem do modelo (0 primeiro, depois 1), usados nos relatórios e gráficos.
nomes_classes = ['Fora do prazo (0)', 'Dentro do prazo (1)']


# ==============================================================================
# 7. AVALIAÇÃO DO MODELO NO CONJUNTO DE TREINAMENTO
# ==============================================================================

# Gera predições do modelo utilizando os próprios dados do conjunto de treino.
y_pred_train = modelo.predict(X_train)

# Imprime o cabeçalho explicativo da acurácia do treino na tela.
print("=== Acurácia no Treino ===")

# Calcula e exibe a acurácia de treino formatada como porcentagem com duas casas decimais.
print(f"Acurácia: {accuracy_score(y_train, y_pred_train):.2%}\n")

# Exibe métricas completas (Precisão, Recall e F1-Score) nos dados de treino.
# zero_division=0 evita aviso quando uma classe nunca é prevista.
print("=== Relatório de Classificação (Treino) ===")
print(classification_report(y_train, y_pred_train, target_names=nomes_classes, zero_division=0))


# ==============================================================================
# 8. PREDIÇÃO E AVALIAÇÃO NO CONJUNTO DE TESTE (DADOS NÃO VISTOS)
# ==============================================================================

# Aplica o modelo treinado para prever as classes das amostras virgens do conjunto de teste (X_test).
y_pred = modelo.predict(X_test)

# Métricas calculadas considerando como "classe de interesse" o pedido FORA do prazo (pos_label=0),
# pois é o atraso que a auditoria quer detectar.
print("=== Métricas no Teste (classe de interesse: FORA do prazo) ===")
print(f"Acurácia : {accuracy_score(y_test, y_pred):.2%}")
print(f"Precisão : {precision_score(y_test, y_pred, pos_label=0, zero_division=0):.2%}")
print(f"Recall   : {recall_score(y_test, y_pred, pos_label=0, zero_division=0):.2%}")
print(f"F1-score : {f1_score(y_test, y_pred, pos_label=0, zero_division=0):.2%}\n")

# Significado de cada métrica neste problema:
# - ACURÁCIA: % de pedidos classificados corretamente (dentro + fora). CUIDADO: se 99% dos pedidos são
#   respondidos no prazo, um modelo que sempre diz "dentro do prazo" já teria 99% e não serviria para nada.
# - PRECISÃO: de todos os pedidos que o modelo ALERTOU como "fora do prazo", quantos realmente atrasaram.
#   Precisão baixa = muitos alarmes falsos (a equipe perde tempo investigando pedidos que estavam bem).
# - RECALL (revocação): de todos os pedidos que REALMENTE atrasaram, quantos o modelo conseguiu pegar.
#   Recall baixo = atrasos passando despercebidos. Para auditoria, costuma ser a métrica mais importante.
# - F1-SCORE: média equilibrada entre precisão e recall; só é alto se os dois forem razoáveis.

# Exibe o relatório com as métricas das duas classes.
print("=== Relatório de Classificação (Teste) ===")
print(classification_report(y_test, y_pred, target_names=nomes_classes, zero_division=0))


# ==============================================================================
# 9. MATRIZ DE CONFUSÃO E REGRAS APRENDIDAS
# ==============================================================================

# Gera a matriz de confusão numérica comparando os rótulos reais (y_test) com os previstos (y_pred).
# labels=[0, 1] garante a ordem fixa: linha/coluna 0 = Fora do prazo, 1 = Dentro do prazo.
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Como ler a matriz (linhas = realidade, colunas = previsão do modelo):
#   [[ acertou "fora"            , previu "dentro" mas era "fora" (ATRASO NÃO DETECTADO) ],
#    [ previu "fora" mas era "dentro" (ALARME FALSO) , acertou "dentro"                   ]]
print("=== Matriz de Confusão (Teste) ===")
print(cm, "\n")

# MUDANÇA EM RELAÇÃO AO CÓDIGO ORIGINAL: os rótulos 'Permaneceu/Cancelou' eram de outro projeto (churn);
# foram trocados pelas classes deste projeto.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomes_classes)
disp.plot(cmap='Blues')
plt.xlabel("Previsão do modelo")     # eixos em português (o padrão do scikit-learn é em inglês)
plt.ylabel("Realidade")
plt.title("Matriz de Confusão - Dados de Teste")
plt.savefig("matriz_confusao.png", dpi=150, bbox_inches='tight')
plt.show()

# Converte a estrutura lógica interna da árvore em texto estruturado por recuos (if/else).
print("=== Regras Aprendidas pelo Modelo ===")
print(export_text(modelo, feature_names=list(X.columns)))


# ==============================================================================
# 10. IMPORTÂNCIA DAS VARIÁVEIS E VISUALIZAÇÃO DA ÁRVORE
# ==============================================================================

# A importância indica o quanto cada variável ajudou a árvore a separar pedidos dentro e fora do prazo
# (valores entre 0 e 1 que somam 1). Variáveis com 0 não foram usadas em nenhuma regra.
importancias = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
print("=== Importância das Variáveis (top 10) ===")
print(importancias.head(10).round(3).to_string(), "\n")

# Gráfico de barras horizontais com as variáveis que a árvore realmente usou.
usadas = importancias[importancias > 0].sort_values()
plt.figure(figsize=(8, 4))
plt.barh(usadas.index, usadas.values, color='steelblue')
plt.xlabel("Importância")
plt.title("Importância das Variáveis - Fatores que mais influenciam o atraso")
plt.tight_layout()
plt.savefig("importancia_variaveis.png", dpi=150, bbox_inches='tight')   # 'tight' evita cortar o título
plt.show()

# Desenha a árvore completa: cada caixa mostra a regra, a quantidade de pedidos e a classe predominante.
# filled=True pinta cada caixa conforme a classe predominante.
plt.figure(figsize=(18, 8))
plot_tree(modelo, feature_names=list(X.columns), class_names=nomes_classes,
          filled=True, rounded=True, fontsize=9)
plt.title("Árvore de Decisão - Prazo de Atendimento (LAI)")
plt.savefig("arvore_decisao.png", dpi=150, bbox_inches='tight')
plt.show()


# ==============================================================================
# 11. PREVISÃO PARA UM PEDIDO NOVO (EXEMPLO)
# ==============================================================================

# Monta um pedido fictício com os mesmos campos usados no treino (valores iguais aos que existem na base).
pedido_novo = pd.DataFrame([{
    'OrgaoDestinatario': df['OrgaoDestinatario'].mode()[0],   # órgão mais frequente da base
    'AssuntoPedido': df['AssuntoPedido'].mode()[0],           # assunto mais frequente da base
    'FoiProrrogado': 'Sim',
    'FoiReencaminhado': 'Não',
    'OrigemSolicitacao': 'Internet',
    'DiasDePrazo': 30,
}])

# Aplica ao pedido novo as MESMAS transformações do treino:
# 1) one-hot; 2) alinha as colunas com as do modelo (reindex preenche com 0 as colunas que não aparecem
# neste único pedido). Obs.: o exemplo usa órgão/assunto frequentes; para um órgão raro, troque o valor
# por 'Outros', como foi feito no treino.
pedido_novo_cod = pd.get_dummies(pedido_novo, drop_first=False, dtype=int)
pedido_novo_cod = pedido_novo_cod.reindex(columns=X.columns, fill_value=0)

# predict_proba devolve a probabilidade de cada classe, na ordem [fora do prazo, dentro do prazo].
prob_fora, prob_dentro = modelo.predict_proba(pedido_novo_cod)[0]
print("=== Previsão para um pedido novo (exemplo) ===")
print(pedido_novo.to_string(index=False))
print(f"Probabilidade de ser atendido DENTRO do prazo: {prob_dentro:.1%}")
print(f"Probabilidade de ser atendido FORA do prazo:   {prob_fora:.1%}\n")


# ==============================================================================
# 12. COMO INTERPRETAR OS RESULTADOS E LIMITAÇÕES
# ==============================================================================
# COMO INTERPRETAR
# - Olhe primeiro a matriz de confusão e o RECALL da classe "Fora do prazo", não a acurácia.
# - As variáveis de maior importância e as regras impressas mostram QUAIS fatores a árvore associou ao atraso.
#   Isso indica associação, não prova de causa: é um ponto de partida para investigar, não uma conclusão.
# - A probabilidade do pedido novo é uma estimativa a partir dos grupos de pedidos parecidos no treino
#   (folhas da árvore). Com árvore rasa, ela assume poucos valores distintos.
#
# LIMITAÇÕES
# 1. DESBALANCEAMENTO EXTREMO: quase todos os pedidos foram respondidos no prazo (apenas alguns atrasos
#    em ~365 registros). No teste (20%) há só 0 ou 1 pedido atrasado, então precisão, recall e F1 da classe
#    "fora do prazo" são praticamente anedóticos: uma mudança de 1 pedido altera tudo.
#    Uma alta acurácia aqui NÃO significa que o modelo detecta atrasos.
# 2. TAMANHO REDUZIDO: 365 registros, de um período curto (janeiro/fevereiro de 2024) e só da esfera federal.
#    O padrão pode não valer para outros períodos ou órgãos.
# 3. TRATAMENTO MANUAL: valores ausentes e inconsistentes foram corrigidos no Excel. Se essa correção alterou
#    datas ou categorias, o rótulo e as regras aprendidas refletem esses ajustes. Convém documentar os critérios.
# 4. VARIÁVEIS: FoiProrrogado só é conhecida depois que o órgão pede mais tempo, e PrazoAtendimento já
#    pode embutir essa prorrogação. Na prática, o modelo é mais confiável para reavaliar pedidos em andamento
#    do que para prever no momento do registro.
# 5. CATEGORIAS RARAS: órgãos e assuntos com menos de 10 pedidos foram agrupados em "Outros".
#
# PRÓXIMOS PASSOS SUGERIDOS (fora do escopo deste modelo didático): ampliar a base com mais meses/anos
# para ter mais exemplos de atraso e usar validação cruzada em vez de um único teste de 20%.
