# ==============================================================================
# PROJETO: ANÁLISE E AUDITORIA DE PRAZOS (Pedidos da LAI - Fala.BR)
# MODELO: ÁRVORE DE DECISÃO (CLASSIFICAÇÃO) - BASE COMPLETA DE 2024 - VERSÃO 2
#
# Antes de rodar, execute o Tratamento_LAI_2024.py (gera o DadosAbertos_LAI_2024_tratado.csv).
#
# DIFERENÇAS EM RELAÇÃO AO Classification_Decision_Tree_2024.py (versão 1):
#   1. DUAS VARIÁVEIS NOVAS sobre o histórico do órgão (seção 4):
#        - HistAtraso: taxa de atraso do órgão nos pedidos que ele JÁ tinha respondido antes do registro do pedido;
#        - Carga30: quantos pedidos o órgão recebeu nos 30 dias anteriores (uma medida de sobrecarga);
#        - MesRegistro: mês em que o pedido foi registrado (sazonalidade).
#   2. SEM class_weight: a v1 dava peso 10 aos atrasos, o que distorcia as probabilidades. Agora o modelo é treinado
#      "normalmente" e suas probabilidades ficam próximas das taxas reais. Quem decide o equilíbrio entre pegar
#      mais atrasos e gerar menos alarmes falsos é o LIMIAR DE CORTE (seção 8), que você pode ajustar.
#   3. NOVA SEÇÃO 8B: tabela mostrando o que acontece com recall, precisão e quantidade de alertas
#      para vários limiares. É ela que responde "qual recall eu consigo e a que preço?".
#   4. NOVA SEÇÃO 10B: ranking dos órgãos com maior taxa de atraso (salvo em CSV) para a auditoria.
#   O restante (alvo, one-hot, divisão 80/20, árvore, pedido novo) segue o mesmo padrão.
# ==============================================================================

# ==============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS
# ==============================================================================

# Path ajuda a localizar o arquivo CSV na mesma pasta deste script, não importa de onde você execute.
from pathlib import Path

# NumPy: usado aqui para contas com datas e para trabalhar com vetores de números.
import numpy as np

# Importa o Pandas para criação, manipulação e estruturação de dados em tabelas (DataFrames).
import pandas as pd

# Importa a função para dividir os dados em conjuntos distintos de treinamento e teste.
from sklearn.model_selection import train_test_split

# Importa a classe do algoritmo de Árvore de Decisão, a função para extrair suas regras em texto
# e a função para desenhar a árvore.
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# Importa funções para calcular métricas de desempenho: acurácia, precisão, recall, F1, AUC e matriz de confusão.
# A AUC mede o quão bem o modelo ORDENA os pedidos do menos para o mais arriscado (0,5 = sorteio; 1,0 = perfeito).
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix, roc_auc_score)

# Importa o módulo PyPlot do Matplotlib para geração e customização de gráficos e figuras.
import matplotlib.pyplot as plt

# Importa a classe do scikit-learn para formatar e desenhar graficamente a matriz de confusão.
from sklearn.metrics import ConfusionMatrixDisplay


# ==============================================================================
# 2. CARREGAMENTO DOS DADOS E CONVERSÃO DAS DATAS
# ==============================================================================

# Monta o caminho do arquivo tratado (na mesma pasta deste script).
pasta = Path(__file__).parent
caminho_csv = pasta / "DadosAbertos_LAI_2024_tratado.csv"

# Lê o CSV tratado (UTF-8 com marca do Excel, por isso 'utf-8-sig').
df = pd.read_csv(caminho_csv, sep=';', encoding='utf-8-sig')

# Converte as colunas de data de texto ("02/01/2024") para o tipo data do Pandas.
for coluna in ['DataRegistro', 'PrazoAtendimento', 'DataResposta']:
    df[coluna] = pd.to_datetime(df[coluna], format='%d/%m/%Y', errors='coerce')

print("=== Dados carregados ===")
print(f"Registros: {len(df)} | Colunas: {df.shape[1]}\n")


# ==============================================================================
# 3. CRIAÇÃO DA VARIÁVEL-ALVO (TARGET)
# ==============================================================================

# Cria a coluna DentroDoPrazo: 1 se a resposta saiu até o prazo (DataResposta <= PrazoAtendimento), 0 se saiu depois.
df['DentroDoPrazo'] = (df['DataResposta'] <= df['PrazoAtendimento']).astype(int)

# Coluna auxiliar: 1 = ATRASOU (o inverso do alvo). Facilita as contas de taxa de atraso mais abaixo.
df['Atrasou'] = 1 - df['DentroDoPrazo']

print("=== Distribuição da variável-alvo ===")
print(df['DentroDoPrazo'].value_counts().rename({1: 'Dentro do prazo (1)', 0: 'Fora do prazo (0)'}))
print()


# ==============================================================================
# 4. SELEÇÃO E TRANSFORMAÇÃO DAS VARIÁVEIS DE ENTRADA (FEATURES)
# ==============================================================================

# Variável 1: quantos dias de prazo o órgão tinha (PrazoAtendimento - DataRegistro).
# Atenção: nos pedidos prorrogados o prazo já inclui a prorrogação (ex.: 20 dias viram 30+).
df['DiasDePrazo'] = (df['PrazoAtendimento'] - df['DataRegistro']).dt.days

# Variável 2 (NOVA): mês do registro, para capturar épocas do ano mais difíceis.
df['MesRegistro'] = df['DataRegistro'].dt.month

# Variáveis 3 e 4 (NOVAS): histórico do órgão. Para cada órgão, olhamos os pedidos dele em ordem de data.
# - HistAtraso: entre os pedidos que o órgão JÁ HAVIA RESPONDIDO antes do registro do pedido atual,
#   que fração foi respondida fora do prazo. Só usa informação que existia na data do registro; usar respostas
#   futuras seria "trapaça" (vazamento de informação), porque na vida real não as teríamos.
# - Carga30: quantos pedidos o órgão recebeu nos 30 dias anteriores a este (sobrecarga de trabalho).
# O código abaixo usa np.searchsorted, que descobre "quantos itens de uma lista ordenada são menores que X".
taxa_geral = df['Atrasou'].mean()                 # taxa de atraso geral, usada quando o órgão não tem histórico ainda
df['HistAtraso'] = taxa_geral
df['Carga30'] = 0
for orgao, grupo in df.groupby('OrgaoDestinatario'):
    # Datas como números inteiros (dias), para poder contar com searchsorted.
    registro = grupo['DataRegistro'].values.astype('datetime64[D]').astype(int)
    resposta = grupo['DataResposta'].values.astype('datetime64[D]').astype(int)
    atraso = grupo['Atrasou'].values

    # HistAtraso: ordena as respostas do órgão por data e acumula os atrasos (soma acumulada).
    ordem = np.argsort(resposta)
    resposta_ord = resposta[ordem]
    atraso_acum = np.cumsum(atraso[ordem])
    # Para cada pedido, quantas respostas do órgão aconteceram ANTES do seu registro?
    qtd_antes = np.searchsorted(resposta_ord, registro, side='left')
    atrasos_antes = np.where(qtd_antes > 0, atraso_acum[np.maximum(qtd_antes - 1, 0)], 0)
    taxa = np.where(qtd_antes > 0, atrasos_antes / np.maximum(qtd_antes, 1), taxa_geral)
    df.loc[grupo.index, 'HistAtraso'] = taxa

    # Carga30: pedidos do órgão registrados entre (data - 30 dias) e a data atual.
    registro_ord = np.sort(registro)
    carga = (np.searchsorted(registro_ord, registro, side='left')
             - np.searchsorted(registro_ord, registro - 30, side='left'))
    df.loc[grupo.index, 'Carga30'] = carga

# Escolha das variáveis de entrada e o motivo de cada exclusão:
# - FICAM: OrgaoDestinatario, AssuntoPedido, FoiProrrogado, FoiReencaminhado, OrigemSolicitacao,
#          DiasDePrazo, MesRegistro, HistAtraso, Carga30.
# - FORA (identificadores): IdPedido e IdSolicitante. São apenas "números de série"; não explicam atraso.
# - FORA (vazamento de informação / data leakage): DataResposta, Decisao e EspecificacaoDecisao só existem
#   DEPOIS que o pedido foi respondido. DataResposta, além disso, é usada para criar o alvo.
# - FORA (pouca informação): Esfera, Situacao e FormaResposta são praticamente constantes após o tratamento.
# - FORA (redundantes): DataRegistro e PrazoAtendimento já estão resumidas em DiasDePrazo e MesRegistro.
colunas_entrada = ['OrgaoDestinatario', 'AssuntoPedido', 'FoiProrrogado', 'FoiReencaminhado',
                   'OrigemSolicitacao', 'DiasDePrazo', 'MesRegistro', 'HistAtraso', 'Carga30']
X = df[colunas_entrada].copy()

# Seleciona a coluna alvo (1 = dentro do prazo, 0 = fora do prazo) e armazena no vetor y.
y = df['DentroDoPrazo']

# Agrupa categorias raras (menos de 500 pedidos) em "Outros", para não criar centenas de colunas.
# Mesmo os órgãos "Outros" continuam diferenciados pela variável HistAtraso, que é numérica.
LIMITE_RARO = 500
for coluna in ['OrgaoDestinatario', 'AssuntoPedido']:
    frequencia = X[coluna].value_counts()
    raras = frequencia[frequencia < LIMITE_RARO].index
    X[coluna] = X[coluna].where(~X[coluna].isin(raras), 'Outros')

# ONE-HOT ENCODING (pd.get_dummies): cada categoria vira uma coluna 0/1. drop_first=True remove a coluna redundante.
X = pd.get_dummies(X, drop_first=True, dtype=int)

print("=== Variáveis de entrada após a transformação ===")
print(f"{X.shape[1]} colunas numéricas\n")


# ==============================================================================
# 5. DIVISÃO DOS DADOS EM TREINO E TESTE
# ==============================================================================

# Divide X e y em treino (80%) e teste (20%), com a mesma proporção de atrasos nos dois (stratify=y).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Treino: {len(X_train)} pedidos ({(y_train == 0).sum()} fora do prazo)")
print(f"Teste:  {len(X_test)} pedidos ({(y_test == 0).sum()} fora do prazo)\n")


# ==============================================================================
# 6. INSTANCIAÇÃO E TREINAMENTO DO MODELO
# ==============================================================================

# Árvore de Decisão:
# - max_depth=4: limita a profundidade para evitar sobreajuste e manter as regras legíveis.
# - min_samples_leaf=100: cada regra final precisa ter pelo menos 100 pedidos (evita regras feitas de acaso).
# - SEM class_weight (diferença da v1): assim a probabilidade que o modelo dá a cada pedido fica próxima da
#   frequência real de atraso de pedidos parecidos. Isso é o que o projeto quer: ESTIMAR PROBABILIDADES.
modelo = DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=42)
modelo.fit(X_train, y_train)

# Nomes das classes na ordem do modelo (0 primeiro, depois 1), usados nos relatórios e gráficos.
nomes_classes = ['Fora do prazo (0)', 'Dentro do prazo (1)']

# Probabilidade de cada pedido de teste ser atendido FORA do prazo (coluna 0 do predict_proba).
prob_fora_teste = modelo.predict_proba(X_test)[:, 0]
prob_fora_treino = modelo.predict_proba(X_train)[:, 0]


# ==============================================================================
# 7. REGRA DE DECISÃO: LIMIAR DE CORTE
# ==============================================================================

# O modelo dá uma PROBABILIDADE de atraso. Para virar um alerta ("este pedido merece atenção"), escolhemos um
# LIMIAR: se a probabilidade de atraso for maior ou igual a ele, o pedido é marcado como "fora do prazo".
# - Limiar BAIXO  -> mais alertas, pega mais atrasos (recall alto), mas muitos alarmes falsos (precisão baixa).
# - Limiar ALTO   -> poucos alertas, mais certeiros (precisão alta), mas deixa passar muitos atrasos.
# Como a taxa de atraso média é ~2,5%, um limiar de 3% já significa "risco acima da média".
# O padrão do scikit-learn seria 50%, o que quase nunca alertaria nada neste problema.
LIMIAR = 0.03

# Converte probabilidades em classes: 0 (alerta de atraso) se prob_fora >= LIMIAR, senão 1 (dentro do prazo).
y_pred_train = np.where(prob_fora_treino >= LIMIAR, 0, 1)
y_pred = np.where(prob_fora_teste >= LIMIAR, 0, 1)


# ==============================================================================
# 8. AVALIAÇÃO DO MODELO NO TREINO E NO TESTE
# ==============================================================================

print("=== Acurácia no Treino ===")
print(f"Acurácia: {accuracy_score(y_train, y_pred_train):.2%}\n")

# Comparar treino e teste ajuda a perceber overfitting: se o treino for MUITO melhor que o teste, o modelo decorou.
print("=== Relatório de Classificação (Treino) ===")
print(classification_report(y_train, y_pred_train, target_names=nomes_classes, zero_division=0))

# Métricas no teste, com "fora do prazo" (pos_label=0) como classe de interesse.
print(f"=== Métricas no Teste (limiar = {LIMIAR:.0%}; classe de interesse: FORA do prazo) ===")
print(f"Acurácia : {accuracy_score(y_test, y_pred):.2%}")
print(f"Precisão : {precision_score(y_test, y_pred, pos_label=0, zero_division=0):.2%}")
print(f"Recall   : {recall_score(y_test, y_pred, pos_label=0, zero_division=0):.2%}")
print(f"F1-score : {f1_score(y_test, y_pred, pos_label=0, zero_division=0):.2%}")
# AUC: independe do limiar. y_test==0 marca os atrasados; prob_fora_teste é o "risco" que o modelo atribuiu.
print(f"AUC      : {roc_auc_score(y_test == 0, prob_fora_teste):.3f}  (0,5 = sorteio; 1,0 = perfeito)\n")

# Significado de cada métrica neste problema:
# - ACURÁCIA: % de pedidos classificados corretamente. CUIDADO: com ~97,5% de pedidos no prazo, dizer
#   "ninguém atrasa" já dá ~97,5% de acurácia sem detectar nada. Por isso ela é a menos útil aqui.
# - PRECISÃO: de todos os pedidos ALERTADOS como "fora do prazo", quantos realmente atrasaram.
#   Precisão baixa = muitos alarmes falsos (a equipe perde tempo olhando pedidos que estavam bem).
# - RECALL (revocação): de todos os pedidos que REALMENTE atrasaram, quantos o modelo alertou.
#   Recall baixo = atrasos passando despercebidos. Em auditoria, perder um atraso costuma custar mais
#   do que olhar um pedido a mais, por isso o recall é o parâmetro principal, MAS sempre junto da
#   % de pedidos alertados (se alertar todos, o recall é 100% e o modelo não serve para nada).
# - F1-SCORE: média equilibrada entre precisão e recall; só é alto se os dois forem razoáveis.
# - AUC: qualidade da ORDENAÇÃO por risco, sem depender do limiar escolhido.

print("=== Relatório de Classificação (Teste) ===")
print(classification_report(y_test, y_pred, target_names=nomes_classes, zero_division=0))


# ==============================================================================
# 8B. TABELA: O PREÇO DE CADA RECALL (VARIANDO O LIMIAR)
# ==============================================================================

# Para vários limiares, mostramos quantos pedidos seriam alertados, quantos atrasos seriam pegos (recall)
# e quantos alertas estariam certos (precisão). Use esta tabela para escolher o limiar do projeto.
print("=== Trade-off por limiar (dados de teste) ===")
print(f"{'Limiar':>7} | {'% alertados':>11} | {'Recall':>7} | {'Precisão':>8}")
total_atrasos = (y_test == 0).sum()
for lim in [0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30]:
    alertas = prob_fora_teste >= lim
    acertos = (alertas & (y_test == 0)).sum()
    recall_lim = acertos / total_atrasos
    precisao_lim = acertos / alertas.sum() if alertas.sum() > 0 else 0
    print(f"{lim:>7.0%} | {alertas.mean():>11.1%} | {recall_lim:>7.1%} | {precisao_lim:>8.1%}")
print()


# ==============================================================================
# 9. MATRIZ DE CONFUSÃO E REGRAS APRENDIDAS
# ==============================================================================

# Matriz de confusão (linhas = realidade, colunas = previsão), na ordem fixa 0 = Fora do prazo, 1 = Dentro do prazo:
#   [[ acertou "fora"            , previu "dentro" mas era "fora" (ATRASO NÃO DETECTADO) ],
#    [ previu "fora" mas era "dentro" (ALARME FALSO) , acertou "dentro"                   ]]
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("=== Matriz de Confusão (Teste) ===")
print(cm, "\n")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomes_classes)
disp.plot(cmap='Blues', values_format='d')
plt.xlabel("Previsão do modelo")     # eixos em português (o padrão do scikit-learn é em inglês)
plt.ylabel("Realidade")
plt.title(f"Matriz de Confusão - Dados de Teste (limiar {LIMIAR:.0%})")
plt.savefig(pasta / "matriz_confusao_2024_v2.png", dpi=150, bbox_inches='tight')
plt.show()

# Converte a estrutura lógica interna da árvore em texto estruturado por recuos (if/else).
print("=== Regras Aprendidas pelo Modelo ===")
print(export_text(modelo, feature_names=list(X.columns)))


# ==============================================================================
# 10. IMPORTÂNCIA DAS VARIÁVEIS E VISUALIZAÇÃO DA ÁRVORE
# ==============================================================================

# Importância: quanto cada variável ajudou a árvore a separar pedidos dentro e fora do prazo (soma 1).
importancias = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
print("=== Importância das Variáveis (top 10) ===")
print(importancias.head(10).round(3).to_string(), "\n")

usadas = importancias[importancias > 0].sort_values()
plt.figure(figsize=(9, 5))
plt.barh(usadas.index, usadas.values, color='steelblue')
plt.xlabel("Importância")
plt.title("Importância das Variáveis - Fatores que mais influenciam o atraso (2024)")
plt.tight_layout()
plt.savefig(pasta / "importancia_variaveis_2024_v2.png", dpi=150, bbox_inches='tight')   # 'tight' evita cortar o título
plt.show()

# Desenha a árvore: cada caixa mostra a regra, a quantidade de pedidos e a classe predominante.
plt.figure(figsize=(34, 11))   # figura larga para as 16 folhas não se sobreporem
plot_tree(modelo, feature_names=list(X.columns), class_names=nomes_classes,
          filled=True, rounded=True, fontsize=8)
plt.title("Árvore de Decisão - Prazo de Atendimento (LAI 2024) - v2")
plt.savefig(pasta / "arvore_decisao_2024_v2.png", dpi=150, bbox_inches='tight')
plt.show()


# ==============================================================================
# 10B. RANKING DE ÓRGÃOS COM MAIOR TAXA DE ATRASO (APOIO À AUDITORIA)
# ==============================================================================

# Resume, por órgão: quantos pedidos recebeu, quantos atrasaram e a taxa de atraso.
# Só entram órgãos com pelo menos 30 pedidos, para a taxa não ser enganosa (1 atraso em 2 pedidos = 50%!).
ranking = df.groupby('OrgaoDestinatario')['Atrasou'].agg(Pedidos='size', Atrasos='sum', TaxaAtraso='mean')
ranking = ranking[ranking['Pedidos'] >= 30].sort_values('TaxaAtraso', ascending=False)
ranking['TaxaAtraso'] = ranking['TaxaAtraso'].round(3)
ranking.to_csv(pasta / "orgaos_taxa_atraso_2024.csv", sep=';', encoding='utf-8-sig')
print("=== 10 órgãos com maior taxa de atraso (mín. 30 pedidos) ===")
print(ranking.head(10).to_string(), "\n")


# ==============================================================================
# 11. PREVISÃO PARA UM PEDIDO NOVO (EXEMPLO)
# ==============================================================================

# Escolhemos o órgão mais frequente e usamos o histórico dele (média de HistAtraso e Carga30 no ano)
# como se fossem os valores do dia em que o pedido novo chegou.
orgao_exemplo = df['OrgaoDestinatario'].mode()[0]
historico_orgao = df[df['OrgaoDestinatario'] == orgao_exemplo]

pedido_novo = pd.DataFrame([{
    'OrgaoDestinatario': orgao_exemplo,
    'AssuntoPedido': df['AssuntoPedido'].mode()[0],           # assunto mais frequente da base
    'FoiProrrogado': 'Sim',
    'FoiReencaminhado': 'Não',
    'OrigemSolicitacao': 'Internet',
    'DiasDePrazo': 30,
    'MesRegistro': 6,
    'HistAtraso': historico_orgao['HistAtraso'].mean(),
    'Carga30': historico_orgao['Carga30'].mean(),
}])

# Aplica as MESMAS transformações do treino: one-hot e alinhamento das colunas com as do modelo
# (reindex preenche com 0 as colunas que não aparecem neste único pedido).
# Obs.: para um órgão/assunto raro, troque o valor por 'Outros', como foi feito no treino.
pedido_novo_cod = pd.get_dummies(pedido_novo, drop_first=False, dtype=int)
pedido_novo_cod = pedido_novo_cod.reindex(columns=X.columns, fill_value=0)

# predict_proba devolve a probabilidade de cada classe, na ordem [fora do prazo, dentro do prazo].
# Como o modelo foi treinado sem pesos, este número é uma estimativa real do risco de atraso.
prob_fora, prob_dentro = modelo.predict_proba(pedido_novo_cod)[0]
print("=== Previsão para um pedido novo (exemplo) ===")
print(pedido_novo.round(3).to_string(index=False))
print(f"Probabilidade de ser atendido DENTRO do prazo: {prob_dentro:.1%}")
print(f"Probabilidade de ser atendido FORA do prazo:   {prob_fora:.1%}")
print(f"Gera alerta (limiar {LIMIAR:.0%})? {'SIM' if prob_fora >= LIMIAR else 'NÃO'}\n")


# ==============================================================================
# 12. COMO INTERPRETAR OS RESULTADOS E LIMITAÇÕES
# ==============================================================================
# COMO INTERPRETAR
# - Use a tabela da seção 8B para escolher o limiar conforme o objetivo: auditoria exaustiva (recall alto, muitos
#   alertas) ou triagem enxuta (poucos alertas, mais certeiros).
# - Recall alto SEM olhar a % de alertados engana: o ganho real do modelo é o recall obtido com poucos alertas.
# - As variáveis de maior importância e as regras impressas mostram QUAIS fatores a árvore associou ao atraso.
#   Isso indica associação, não prova de causa.
#
# LIMITAÇÕES
# 1. DESBALANCEAMENTO: só ~2,5% dos pedidos atrasaram. Com recall alto, a maioria dos alertas será falso alarme.
# 2. VARIÁVEIS LIMITADAS: o modelo não conhece a complexidade real do pedido, a equipe do órgão ou o texto do pedido.
#    Isso limita o desempenho mais do que o algoritmo; até modelos mais complexos teriam resultado parecido.
# 3. FoiProrrogado e DiasDePrazo só são conhecidos depois que o órgão pede mais tempo (o prazo já inclui a
#    prorrogação). Logo, o modelo serve melhor para reavaliar pedidos em andamento do que para prever no registro.
# 4. HISTÓRICO NO INÍCIO DO ANO: em janeiro o HistAtraso ainda tem poucas respostas anteriores (usa a taxa
#    geral quando não há nenhuma), então é menos confiável nos primeiros meses.
# 5. UM ANO, SÓ FEDERAL: dados de 2024 e esfera Federal. Outros anos e esferas podem ter padrões diferentes.
# 6. UM ÚNICO TESTE: o resultado vem de uma divisão aleatória 80/20. Para mais confiança, o ideal seria testar
#    em meses posteriores aos do treino (validação temporal) e usar validação cruzada.
