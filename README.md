# Construindo um Tokenizador BPE e Explorando o WordPiece

## Descrição do Projeto

Este laboratório implementa os fundamentos do algoritmo **Byte Pair Encoding (BPE)** do zero e explora o funcionamento do algoritmo **WordPiece** através da biblioteca Hugging Face Transformers. O objetivo é compreender como os modelos de linguagem modernos processam texto através de sub-palavras.

## Objetivos de Aprendizado

1. **Entender o BPE**: Implementar o motor de frequências e o loop de fusão
2. **Aplicar a Tokenização**: Usar o tokenizador multilíngue BERT para texto em português
3. **Analisar Sub-palavras**: Compreender o papel dos símbolos `##` e a robustez contra vocabulário desconhecido

## Estrutura do Repositório

```
├── tokenizador_bpe_wordpiece.ipynb   # Notebook com todas as implementações
├── README.md                           # Este arquivo
└── .git/                              # Controle de versão Git
```

## Como Executar

### Pré-requisitos
- Python 3.7+
- Jupyter Notebook
- Biblioteca `transformers` (será instalada automaticamente no notebook)

### Passos para Execução

1. **Clone o repositório** (se aplicável):
   ```bash
   git clone <repository-url>
   cd Construindo-um-Tokenizador-BPE-e-Explorando-o-WordPiece
   ```

2. **Abra o Jupyter Notebook**:
   ```bash
   jupyter notebook tokenizador_bpe_wordpiece.ipynb
   ```

3. **Execute as células** sequencialmente (Shift + Enter em cada célula)

## Conteúdo Detalhado

### Tarefa 1: O Motor de Frequências

**Objetivo**: Implementar a função `get_stats(vocab)` que calcula as frequências de pares adjacentes.

**Corpus de Treinamento**:
```python
vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}
```

**Validação**: O par `('e', 's')` deve retornar frequência **9** (6 de "newest" + 3 de "widest")

**Função Implementada**:
```python
def get_stats(vocab):
    pairs = {}
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs
```

### Tarefa 2: O Loop de Fusão

**Objetivo**: Implementar a função `merge_vocab()` e executar 5 iterações do BPE.

**Função de Fusão**:
```python
def merge_vocab(pair, v_in):
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in v_in.items():
        new_word = word.replace(bigram, replacement)
        v_out[new_word] = freq
    return v_out
```

**Resultado das Iterações**:
A cada iteração, o algoritmo:
1. Encontra o par mais frequente
2. Realiza a fusão (merge) desse par
3. Atualiza o vocabulário

Após 5 iterações, você observará a formação de tokens morfológicos como:
- `est</w>` (sufixo de superlativos)
- `er</w>` (sufixo de comparativos)
- `w</w>` (final de palavra)

### Tarefa 3: WordPiece e Hugging Face

**Objetivo**: Explorar o tokenizador WordPiece do BERT em texto português.

#### Tokenizador Utilizado
- **Modelo**: `bert-base-multilingual-cased`
- **Tipo**: WordPiece com suporte a múltiplos idiomas
- **Características**: Preserva maiúsculas, sup... palavras raras via sub-palavras

#### Frase de Teste
```
"Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."
```

#### Explicação do WordPiece

##### O que significam os símbolos `##`?

Os símbolos duplo-cerquilha (`##`) indicam que o token é uma **continuação de uma palavra anterior**:

- **Tokens sem ##**: Início de palavra ou palavra completa
  - `Os`, `hiper`, `transformer`, `são`

- **Tokens com ##**: Continuação de palavra
  - `##mente` → continuação de palavra anterior
  - `##mente` em `inconstitucionalmente` → parte de uma palavra maior

**Exemplo de Decomposição**:
```
"inconstitucionalmente"
    |
[incons] [##tituc] [##ion] [##al] [##mente]
```

##### Por que isso previne falhas com vocabulário desconhecido?

1. **Robustez contra palavras raras**: Mesmo palabras nunca vistas no treinamento podem ser decompostas em sub-palavras conhecidas

2. **Processamento por contexto**: Cada sub-palavra passa pela rede Transformer, que usa o contexto da frase para inferir significado

3. **Vocabulário otimizado**: Em vez de manter 1 milhão de palavras, o modelo usa ~30.000 tokens, cobrindo praticamente qualquer palavra possível

4. **Transferência de conhecimento**: Sub-palavras com padrões semelhantes compartilham informação
   - `mente` (sufixo de advérbio) aparece em muitas palavras
   - O modelo aprende o significado desse padrão uma vez e aplica a múltiplas palavras

**Vantagens do WordPiece**:
- Sem "tokens desconhecidos" (`[UNK]`)
- Tamanho de vocabulário controlado
- Melhor representação de morfologia em português
- Generalização efetiva para novas palavras

## Usando o Tokenizador

```python
from transformers import AutoTokenizer

# Carregar tokenizador
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Tokenizar texto
tokens = tokenizer.tokenize("sua frase aqui")
print(tokens)

# Converter para IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)
```

## Observações Importantes

### Sobre o BPE Implementado
- O BPE é um algoritmo **guloso** (greedy)
- Sempre escolhe o par mais frequente para fusão
- Produz tokens que aparecem frequentemente juntos no corpus
- Comum em modelos como GPT, GPT-2, e variantes

### Sobre o WordPiece
- Ligeiramente diferente do BPE: usa probabilidade em vez de frequência
- Prioriza fusões que melhoram a pontuação geral do modelo (likelihood)
- Usado em BERT, RoBERTa e outros modelos baseados em BERT
- Mais sofisticado que o BPE simples

## Uso de IA Generativa

Este projeto foi desenvolvido com suporte de IA generativa (GitHub Copilot) e revisado por **Wendril Gabriel**.

**Trechos gerados e revisados**:
- Função `merge_vocab()`: Estrutura base revisada para garantir funcionalidade correta
- Função `get_stats()`: Lógica de iteração sobre símbolos e contagem de pares
- Loop principal: Estrutura de iteração e formatação de output
- Documentação e comentários: Explicações técnicas expandidas

**Revisor**: Wendril Gabriel

**Integridade Acadêmica**: Todos os trechos foram testados, validados e compreendidos antes da submissão. O código foi verificado mostrando a validação correta do par `('e', 's')` com frequência 9.

## Referências

- Sennrich et al. (2016) - "Neural Machine Translation of Rare Words with Subword Units" (BPE Original)
- Devlin et al. (2019) - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- Hugging Face Documentation: https://huggingface.co/

## Versão

- **Versão**: 1.0
- **Data**: 2026-03
- **Tag Git**: v1.0
