# ==============================================================================
# PROJETO: ANÁLISE E AUDITORIA DE PRAZOS (Pedidos da LAI - Fala.BR)
# MODELO: ÁRVORE DE DECISÃO (CLASSIFICAÇÃO) - BASE COMPLETA DE 2024
#
# Este script é a versão "base completa" do Classification_Decision_Tree.py.
# Antes de rodar, execute o Tratamento_LAI_2024.py para gerar o arquivo DadosAbertos_LAI_2024_tratado.csv.
#
# DIFERENÇAS EM RELAÇÃO AO Classification_Decision_Tree.py (base de 365 registros):
#   1. ARQUIVO: lê DadosAbertos_LAI_2024_tratado.csv (~117 mil pedidos federais) em vez de DadosAbertos_LAI.csv.
#      O arquivo agora é UTF-8 (gerado pelo Tratamento_LAI_2024.py), então o encoding mudou de 'cp1252' para 'utf-8-sig'.
#   2. ATRASOS SUFICIENTES: antes eram 3 pedidos fora do prazo; agora são milhares (~2,5%). O teste (20%) tem
#      centenas de atrasos, então precisão, recall e F1 passam a ter significado estatístico.
#   3. CATEGORIAS RARAS: com muito mais pedidos, o limite para agrupar em "Outros" subiu de 10 para 500 pedidos.
#   4. PROFUNDIDADE: max_depth passou de 3 para 4, pois há dados suficientes para uma regra a mais sem decorar.
#   5. ARQUIVOS DE SAÍDA: as imagens ganham o sufixo _2024 para não sobrescrever as da versão anterior.
#   6. LIMITAÇÕES (seção 12): reescritas, pois o problema agora não é falta de dados, e sim o desbalanceamento.
#   O restante (alvo, variáveis, one-hot, métricas, árvore, pedido novo) segue o mesmo padrão.
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

# Importa a classe do algoritmo de Árvore de Decisão, a função para extrair suas regras em texto
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

# Monta o caminho do arquivo tratado (na mesma pasta deste script).
caminho_csv = Path(__file__).parent / "DadosAbertos_LAI_2024_tratado.csv"

# Lê o CSV tratado. Diferença da versão anterior: encoding='utf-8-sig', pois o arquivo foi gravado em UTF-8
# pelo Tratamento_LAI_2024.py (o 'sig' remove uma marca invisível que o Excel usa no início do arquivo).
df = pd.read_csv(caminho_csv, sep=';', encoding='utf-8-sig')

# Não precisamos mais remover linhas vazias: o tratamento já fez essa limpeza.

# Converte as colunas de data de texto ("02/01/2024") para o tipo data do Pandas.
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
df['DentroDoPrazo'] = (df['DataResposta'] <= df['PrazoAtendimento']).astype(int)

# Mostra quantos pedidos há em cada classe. A base ainda é desbalanceada (poucos atrasos), mas agora são milhares.
print("=== Distribuição da variável-alvo ===")
print(df['DentroDoPrazo'].value_counts().rename({1: 'Dentro do prazo (1)', 0: 'Fora do prazo (0)'}))
print()


# ==============================================================================
# 4. SELEÇÃO E TRANSFORMAÇÃO DAS VARIÁVEIS DE ENTRADA (FEATURES)
# ==============================================================================

# Cria uma variável numérica nova: quantos dias de prazo o órgão tinha (PrazoAtendimento - DataRegistro).
# Atenção: nos pedidos prorrogados o prazo já inclui a prorrogação (ex.: 20 dias viram 30+).
df['DiasDePrazo'] = (df['PrazoAtendimento'] - df['DataRegistro']).dt.days

# Escolha das variáveis de entrada (mesmas da versão anterior) e o motivo de cada exclusão:
# - FICAM: OrgaoDestinatario, AssuntoPedido, FoiProrrogado, FoiReencaminhado, OrigemSolicitacao, DiasDePrazo.
# - FORA (identificadores): IdPedido e IdSolicitante. São apenas "números de série"; não explicam atraso.
# - FORA (vazamento de informação / data leakage): DataResposta, Decisao e EspecificacaoDecisao só existem
#   DEPOIS que o pedido foi respondido. DataResposta, além disso, é usada para criar o alvo.
# - FORA (pouca informação): Esfera, Situacao e FormaResposta são praticamente constantes após o tratamento.
# - FORA (redundantes): DataRegistro e PrazoAtendimento já estão resumidas em DiasDePrazo.
colunas_entrada = ['OrgaoDestinatario', 'AssuntoPedido', 'FoiProrrogado',
                   'FoiReencaminhado', 'OrigemSolicitacao', 'DiasDePrazo']
X = df[colunas_entrada].copy()

# Seleciona a coluna alvo (1 = dentro do prazo, 0 = fora do prazo) e armazena no vetor y.
y = df['DentroDoPrazo']

# Existem centenas de órgãos e assuntos diferentes; muitos aparecem poucas vezes.
# Para não criar centenas de colunas, agrupamos as categorias raras (menos de 500 pedidos) em "Outros".
# (Na versão anterior o limite era 10; com mais dados, o limite pode ser maior.)
LIMITE_RARO = 500
for coluna in ['OrgaoDestinatario', 'AssuntoPedido']:
    frequencia = X[coluna].value_counts()
    raras = frequencia[frequencia < LIMITE_RARO].index
    X[coluna] = X[coluna].where(~X[coluna].isin(raras), 'Outros')

# Transformação das categorias em números pelo método ONE-HOT ENCODING (pd.get_dummies):
# cada categoria vira uma coluna própria com valor 0 ou 1. Exemplo: FoiProrrogado vira "FoiProrrogado_Sim".
# Isso evita numerar as categorias (1, 2, 3...), o que criaria uma falsa ordem entre elas.
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
# - stratify=y: Preserva a mesma proporção de pedidos dentro/fora do prazo no treino e no teste.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Treino: {len(X_train)} pedidos ({(y_train == 0).sum()} fora do prazo)")
print(f"Teste:  {len(X_test)} pedidos ({(y_test == 0).sum()} fora do prazo)\n")


# ==============================================================================
# 6. INSTANCIAÇÃO E TREINAMENTO DO MODELO
# ==============================================================================

# Cria o modelo de Árvore de Decisão:
# - max_depth=4: limita a profundidade a 4 níveis para evitar sobreajuste e manter as regras legíveis
#   (era 3; com mais dados cabe um nível a mais).
# - class_weight={0: 10, 1: 1}: como os pedidos fora do prazo (classe 0) são minoria, cada um deles "vale" 10 no treino,
#   para a árvore não aprender apenas "todo mundo responde no prazo". Testamos também class_weight='balanced'
#   (peso ~40): ele pega mais atrasos (recall 91%), mas marca 40% de TODOS os pedidos como atrasados, o que
#   inviabiliza a auditoria. O peso 10 é um meio-termo: alerta ~7% dos pedidos. Você pode testar outros valores.
# - min_samples_leaf=100: cada regra final precisa ter pelo menos 100 pedidos. Evita regras baseadas em
#   pouquíssimos casos (acaso).
# - random_state=42: resultado reproduzível.
modelo = DecisionTreeClassifier(max_depth=4, class_weight={0: 10, 1: 1},
                                min_samples_leaf=100, random_state=42)

# Treina o algoritmo aprendendo as regras de divisão nos dados de treino (X_train e y_train).
modelo.fit(X_train, y_train)

# Nomes das classes na ordem do modelo (0 primeiro, depois 1), usados nos relatórios e gráficos.
nomes_classes = ['Fora do prazo (0)', 'Dentro do prazo (1)']


# ==============================================================================
# 7. AVALIAÇÃO DO MODELO NO CONJUNTO DE TREINAMENTO
# ==============================================================================

# Gera predições do modelo utilizando os próprios dados do conjunto de treino.
y_pred_train = modelo.predict(X_train)

print("=== Acurácia no Treino ===")
print(f"Acurácia: {accuracy_score(y_train, y_pred_train):.2%}\n")

# Exibe métricas completas (Precisão, Recall e F1-Score) nos dados de treino.
# Comparar treino e teste ajuda a perceber overfitting: se o treino for MUITO melhor que o teste, o modelo decorou.
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
# - ACURÁCIA: % de pedidos classificados corretamente. CUIDADO: se ~97% dos pedidos são respondidos no prazo,
#   um modelo que sempre diz "dentro do prazo" já teria ~97% e não serviria para nada.
# - PRECISÃO: de todos os pedidos que o modelo ALERTOU como "fora do prazo", quantos realmente atrasaram.
#   Precisão baixa = muitos alarmes falsos (a equipe perde tempo investigando pedidos que estavam bem).
# - RECALL (revocação): de todos os pedidos que REALMENTE atrasaram, quantos o modelo conseguiu pegar.
#   Recall baixo = atrasos passando despercebidos. Para auditoria, costuma ser a métrica mais importante.
# - F1-SCORE: média equilibrada entre precisão e recall; só é alto se os dois forem razoáveis.

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

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomes_classes)
disp.plot(cmap='Blues', values_format='d')
plt.xlabel("Previsão do modelo")     # eixos em português (o padrão do scikit-learn é em inglês)
plt.ylabel("Realidade")
plt.title("Matriz de Confusão - Dados de Teste (2024)")
plt.savefig("matriz_confusao_2024.png", dpi=150, bbox_inches='tight')
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
plt.figure(figsize=(9, 5))
plt.barh(usadas.index, usadas.values, color='steelblue')
plt.xlabel("Importância")
plt.title("Importância das Variáveis - Fatores que mais influenciam o atraso (2024)")
plt.tight_layout()
plt.savefig("importancia_variaveis_2024.png", dpi=150, bbox_inches='tight')   # 'tight' evita cortar o título
plt.show()

# Desenha a árvore completa: cada caixa mostra a regra, a quantidade de pedidos e a classe predominante.
plt.figure(figsize=(34, 11))   # figura larga para as 16 folhas não se sobreporem
plot_tree(modelo, feature_names=list(X.columns), class_names=nomes_classes,
          filled=True, rounded=True, fontsize=8)
plt.title("Árvore de Decisão - Prazo de Atendimento (LAI 2024)")
plt.savefig("arvore_decisao_2024.png", dpi=150, bbox_inches='tight')
plt.show()


# ==============================================================================
# 11. PREVISÃO PARA UM PEDIDO NOVO (EXEMPLO)
# ==============================================================================

# Monta um pedido fictício com os mesmos campos usados no treino (valores que existem na base).
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
# Atenção: com o peso extra para o atraso essa probabilidade fica "inflada" para o atraso; use-a como
# um índice de risco para comparar pedidos, não como a chance real de atraso.
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
# - Há um equilíbrio entre os dois erros: aumentar o recall (pegar mais atrasos) costuma gerar mais alarmes falsos
#   (precisão menor). A escolha depende do custo: investigar um falso alarme é barato perto de perder um prazo?
# - As variáveis de maior importância e as regras impressas mostram QUAIS fatores a árvore associou ao atraso.
#   Isso indica associação, não prova de causa: é um ponto de partida para investigar.
#
# LIMITAÇÕES
# 1. DESBALANCEAMENTO: só ~2,5% dos pedidos atrasaram. Mesmo com o peso extra, é difícil prever atrasos raros:
#    espere precisão baixa (a maioria dos alertas será falso alarme) e recall moderado. O modelo serve para
#    PRIORIZAR a fila de verificação (risco maior que a média), não para decidir sozinho.
# 2. VARIÁVEIS LIMITADAS: o modelo usa só informações do pedido (órgão, assunto, prazo...). Não conhece a carga de
#    trabalho do órgão, o tamanho da equipe ou a complexidade real da resposta, que provavelmente explicam o atraso.
# 3. FoiProrrogado só é conhecido depois que o órgão pede mais tempo, e PrazoAtendimento já inclui a prorrogação.
#    Logo, o modelo serve melhor para reavaliar pedidos em andamento do que para prever no momento do registro.
# 4. UM ANO, SÓ FEDERAL: dados de 2024 e esfera Federal. Estados, municípios e outros anos podem ter padrões diferentes.
# 5. TRATAMENTO: removemos pedidos sem resposta (~400) e outras esferas, e trocamos "?" por "-" nos nomes de órgãos.
#    Esses critérios estão registrados no Tratamento_LAI_2024.py.
# 6. UM ÚNICO TESTE: o resultado vem de uma divisão 80/20. Para mais confiança, o próximo passo seria usar
#    validação cruzada (fora do escopo deste modelo didático).
