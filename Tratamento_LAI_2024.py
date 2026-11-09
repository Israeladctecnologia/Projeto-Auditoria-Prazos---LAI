# ==============================================================================
# PROJETO: ANÁLISE E AUDITORIA DE PRAZOS (Pedidos da LAI - Fala.BR)
# TRATAMENTO DOS DADOS (ETL) DO ARQUIVO ORIGINAL DE 2024
#
# O que este script faz: pega o arquivo bruto "20261104_Pedidos_csv_2024.csv" (todos os pedidos de 2024)
# e faz em Python o mesmo trabalho que foi feito à mão no Excel para o "DadosAbertos_LAI.csv":
#   limpar caracteres e separadores, reduzir colunas, tratar valores ausentes e inconsistentes.
# O resultado é salvo em outro arquivo ("DadosAbertos_LAI_2024_tratado.csv"), que será usado pelo
# script Classification_Decision_Tree_2024.py. O arquivo original NUNCA é alterado.
#
# Isto é a parte "T" do ETL (Extract, Transform, Load): extrair -> transformar -> carregar.
# ==============================================================================

# ==============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS
# ==============================================================================

# Path ajuda a localizar os arquivos na mesma pasta deste script, não importa de onde você execute.
from pathlib import Path

# Importa o Pandas para manipulação de dados em tabelas (DataFrames).
import pandas as pd


# ==============================================================================
# 2. EXTRAÇÃO: LEITURA DO ARQUIVO BRUTO
# ==============================================================================

# Monta os caminhos do arquivo de entrada (bruto) e do arquivo de saída (tratado).
pasta = Path(__file__).parent
arquivo_bruto = pasta / "20261104_Pedidos_csv_2024.csv"
arquivo_tratado = pasta / "DadosAbertos_LAI_2024_tratado.csv"

# Lê o CSV bruto. Três cuidados que este arquivo exige (são os "caracteres especiais e separadores" do Excel):
# - sep=';'        -> as colunas são separadas por ponto e vírgula;
# - encoding='cp850' -> o arquivo foi gerado no padrão antigo do Windows/DOS. Com UTF-8 daria erro e com
#                       cp1252 os acentos sairiam trocados. O cp850 foi o que devolveu "Concluída", "Não" etc.;
# - quoting=3      -> cada linha inteira vem entre aspas ("...;...;..."), o que confunde o leitor.
#                     Com quoting=3 ele ignora as aspas e depois removemos as que sobram nas pontas.
# - dtype=str      -> lê tudo como texto, para o Pandas não "adivinhar" tipos errados (ex.: apagar zeros à esquerda).
df = pd.read_csv(arquivo_bruto, sep=';', encoding='cp850', quoting=3, dtype=str, low_memory=False)
linhas_brutas = len(df)

# Remove as aspas que sobraram no início do primeiro cabeçalho/valor e no fim do último.
df.columns = [coluna.strip('"') for coluna in df.columns]
df['IdPedido'] = df['IdPedido'].str.lstrip('"')

print("=== Arquivo bruto ===")
print(f"Linhas: {linhas_brutas} | Colunas: {df.shape[1]}\n")


# ==============================================================================
# 3. TRANSFORMAÇÃO 1: REDUÇÃO DE COLUNAS
# ==============================================================================

# O arquivo bruto tem 24 colunas (a última é vazia, criada por um ';' extra no fim de cada linha).
# Mantemos as mesmas 15 colunas do DadosAbertos_LAI.csv que você tratou no Excel.
# Ficaram de fora: ProtocoloPedido (outro identificador), UF e Municipio (só preenchidos para órgãos
# estaduais/municipais), SubAssuntoPedido e Tag (quase sempre vazias), DetalhamentoDecisao,
# MotivoNegativaAcesso e PrazoRestricaoAcesso (texto livre/vazias).
colunas_mantidas = ['IdPedido', 'Esfera', 'OrgaoDestinatario', 'Situacao', 'DataRegistro',
                    'PrazoAtendimento', 'FoiProrrogado', 'FoiReencaminhado', 'FormaResposta',
                    'OrigemSolicitacao', 'IdSolicitante', 'AssuntoPedido', 'DataResposta',
                    'Decisao', 'EspecificacaoDecisao']
df = df[colunas_mantidas].copy()


# ==============================================================================
# 4. TRANSFORMAÇÃO 2: LIMPEZA DE TEXTO E VALORES AUSENTES
# ==============================================================================

# Remove espaços sobrando no início e no fim de todos os textos (ex.: "CBTU ... Urbanos " vira "CBTU ... Urbanos").
for coluna in df.columns:
    df[coluna] = df[coluna].str.strip()

# No arquivo bruto, o traço dos nomes de órgãos virou "?" (ex.: "CBTU ? Companhia Brasileira...").
# É um caractere que se perdeu na gravação original; trocamos pelo hífen, como no seu arquivo do Excel.
df['OrgaoDestinatario'] = df['OrgaoDestinatario'].str.replace(' ? ', ' - ', regex=False)

# Campos que vêm vazios (ou só com espaço) significam "sem informação". Em vez de deixar o vazio,
# preenchemos com um texto claro, para o modelo e para quem ler o arquivo.
# - FormaResposta e Decisao: poucos vazios, viram "Não informado";
# - EspecificacaoDecisao: vazia quando não há detalhe, vira "Não informado" também.
for coluna in ['FormaResposta', 'Decisao', 'EspecificacaoDecisao']:
    df[coluna] = df[coluna].replace('', pd.NA).fillna('Não informado')


# ==============================================================================
# 5. TRANSFORMAÇÃO 3: CONVERSÃO DE DATAS E REMOÇÃO DE REGISTROS INVÁLIDOS
# ==============================================================================

# Converte as colunas de data de texto ("02/01/2024") para o tipo data.
# errors='coerce' transforma datas inválidas ou vazias em vazio (NaT) em vez de parar o programa.
for coluna in ['DataRegistro', 'PrazoAtendimento', 'DataResposta']:
    df[coluna] = pd.to_datetime(df[coluna], format='%d/%m/%Y', errors='coerce')

# Guarda quantas linhas temos para mostrar o efeito de cada filtro (rastreabilidade para a auditoria).
print("=== Filtros aplicados ===")

# FILTRO A - Esfera Federal: o projeto foca em órgãos federais (como na base anterior).
# Esferas estaduais e municipais têm regras e realidades diferentes e misturá-las atrapalharia o modelo.
antes = len(df)
df = df[df['Esfera'] == 'Federal']
print(f"Removidos por não serem esfera Federal: {antes - len(df)}")

# FILTRO B - Sem DataResposta: pedido ainda sem resposta (cadastrado, prorrogado ou encaminhado)
# não permite saber se foi dentro ou fora do prazo, ou seja, não tem rótulo para o modelo aprender.
antes = len(df)
df = df.dropna(subset=['DataResposta'])
print(f"Removidos por não terem DataResposta: {antes - len(df)}")

# FILTRO C - Datas inconsistentes: resposta antes do registro, ou prazo igual/anterior ao registro, não fazem sentido.
antes = len(df)
df = df[(df['DataResposta'] >= df['DataRegistro']) & (df['PrazoAtendimento'] > df['DataRegistro'])]
print(f"Removidos por datas inconsistentes: {antes - len(df)}")

# FILTRO D - Duplicados: o mesmo IdPedido não pode aparecer duas vezes.
antes = len(df)
df = df.drop_duplicates(subset='IdPedido')
print(f"Removidos por IdPedido duplicado: {antes - len(df)}\n")

# Observação sobre prazos muito longos (até 866 dias): foram MANTIDOS. São poucos (cerca de 170 pedidos com mais de 60 dias) e
# a Árvore de Decisão não é sensível a valores extremos, mas vale lembrar deles ao interpretar os resultados.


# ==============================================================================
# 6. CARREGAMENTO: GRAVAÇÃO DO ARQUIVO TRATADO E RESUMO
# ==============================================================================

# Volta as datas para o formato dia/mês/ano, igual ao arquivo do Excel, para ficar fácil de conferir.
for coluna in ['DataRegistro', 'PrazoAtendimento', 'DataResposta']:
    df[coluna] = df[coluna].dt.strftime('%d/%m/%Y')

# Salva o arquivo tratado:
# - sep=';' e index=False -> mesmo padrão do arquivo anterior, sem coluna de número de linha;
# - encoding='utf-8-sig'  -> UTF-8 com marcação que faz o Excel abrir os acentos corretamente.
df.to_csv(arquivo_tratado, sep=';', index=False, encoding='utf-8-sig')

# Resumo final para conferência.
print("=== Resultado ===")
print(f"Linhas brutas: {linhas_brutas} -> Linhas tratadas: {len(df)}")
print(f"Arquivo gravado em: {arquivo_tratado.name}")
print("Valores ausentes por coluna (deve ser tudo 0):")
print(df.isna().sum().to_string())
