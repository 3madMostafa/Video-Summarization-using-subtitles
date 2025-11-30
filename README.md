# YouTube Video Summarization: A Comparative Study

## Overview

This project implements and compares multiple text summarization approaches on YouTube video transcripts. It evaluates five different vectorization methods combined with graph-based ranking algorithms to generate extractive summaries and compares their performance using ROUGE metrics.

## Dataset

The project uses a dataset of MrBeast YouTube video transcripts (`mr_beast.csv`) containing:
- **523 videos** with original subtitles
- Reference summaries for evaluation
- Two columns: `text` (full transcript) and `summary` (reference summary)

## Methodology

### Text Preprocessing Pipeline

1. **Cleaning**
   - Lowercase conversion
   - Punctuation removal (except periods)
   - Digit removal
   - Whitespace normalization

2. **Sentence Segmentation**
   - NLTK's sentence tokenizer

3. **Tokenization**
   - Word-level tokenization for vector processing

### Summarization Approaches

The project implements **five different vectorization techniques**:

#### 1. **TF-IDF + PageRank**
- Uses Term Frequency-Inverse Document Frequency for sentence vectorization
- Applies PageRank algorithm for sentence ranking
- Creates similarity matrix using cosine similarity
- Selects top-k sentences based on PageRank scores

#### 2. **Doc2Vec + HITS**
- Employs Doc2Vec embeddings (100-dimensional vectors)
- Trains on the document corpus
- Uses HITS (Hyperlink-Induced Topic Search) algorithm for ranking
- Leverages authority scores for sentence selection

#### 3. **T5 Transformer**
- Leverages pre-trained T5-base model
- Generates abstractive summaries
- Summary length: 80-100 tokens
- Uses sequence-to-sequence architecture

#### 4. **SentenceTransformer + HITS**
- Uses 'all-MiniLM-L6-v2' pre-trained model
- Generates high-quality sentence embeddings
- Applies HITS algorithm for extractive summarization
- Best overall performance in evaluation

#### 5. **FastText + HITS**
- Trains custom FastText embeddings (128-dimensional)
- Window size: 5, Min count: 3
- Skip-gram architecture (sg=1)
- Combines with HITS for sentence selection

### Ranking Algorithms

- **PageRank**: Measures sentence importance based on similarity connections in the graph
- **HITS**: Distinguishes between hub and authority scores, using authority scores for ranking

## Installation

### Requirements

```bash
pip install rouge
pip install sentence-transformers
pip install nltk
pip install gensim
pip install spacy
pip install scikit-learn
pip install pandas
pip install numpy
pip install networkx
pip install transformers
```

### NLTK Data

```python
import nltk
nltk.download('punkt')
```

## Usage

### Basic Implementation

```python
import pandas as pd

# Load and clean data
data = pd.read_csv("mr_beast.csv")
data["text"] = data["text"].apply(lambda x: clean_text(x.split()))
data["summary"] = data["summary"].apply(lambda x: clean_text(x.split()))

# Generate summaries using different methods
# TF-IDF approach
data['summary_tfidf'] = data.apply(lambda row: tfidf_summary(row['text'], 5), axis=1)

# Doc2Vec approach
data['summary_doc2vec'] = data.apply(lambda row: doc2vec_summary(row['text'], 5), axis=1)

# T5 Transformer approach
data['t5_summary'] = data.apply(lambda row: vectorize_sentences3(row['text']), axis=1)

# SentenceTransformer approach
data['transformer_summary'] = data.apply(lambda row: transformer_summrizer(row['text']), axis=1)

# FastText approach
data['fast_text_summary'] = data.apply(lambda row: fast_text_summary(row['text']), axis=1)
```

### Individual Summarization Functions

```python
# Example: Summarize a single text using TF-IDF
text = "Your long text here..."
summary = tfidf_summary(text, summary_size=5)
print(summary)

# Example: Summarize using SentenceTransformer
summary = transformer_summrizer(text)
print(summary)
```

## Results

Performance comparison using ROUGE metrics (average across 517 videos after filtering):

### ROUGE-1 Scores (Unigram Overlap)

| Method | F1 Score | Precision | Recall |
|--------|----------|-----------|--------|
| TF-IDF | 0.278 | 0.325 | 0.275 |
| Doc2Vec | 0.325 | 0.421 | 0.277 |
| T5 | **0.371** | 0.350 | **0.424** |
| SentenceTransformer | **0.391** | **0.437** | 0.381 |
| FastText | 0.388 | **0.464** | 0.355 |

### ROUGE-2 Scores (Bigram Overlap)

| Method | F1 Score | Precision | Recall |
|--------|----------|-----------|--------|
| TF-IDF | 0.138 | 0.178 | 0.120 |
| Doc2Vec | 0.142 | 0.194 | 0.119 |
| T5 | 0.197 | 0.191 | **0.217** |
| SentenceTransformer | **0.207** | **0.254** | 0.185 |
| FastText | 0.194 | 0.251 | 0.169 |

### Key Findings

1. **SentenceTransformer** achieved the highest overall performance:
   - ROUGE-1 F1: 0.391
   - ROUGE-2 F1: 0.207
   - Best balance between precision and recall

2. **T5** showed strong recall performance:
   - Highest ROUGE-1 recall (0.424)
   - Captures more reference content
   - Abstractive approach provides paraphrasing

3. **FastText** demonstrated competitive precision:
   - Highest ROUGE-1 precision (0.464)
   - Lower recall suggests more selective summarization

4. **Traditional methods** (TF-IDF, Doc2Vec):
   - Performed reasonably well
   - Lower scores than transformer-based approaches
   - More computationally efficient

5. **Doc2Vec** outperformed TF-IDF:
   - Better semantic understanding
   - Higher F1 scores across both metrics

## Project Structure

```
├── mr_beast.csv                 # Input dataset (523 videos)
├── summarization.ipynb          # Main implementation notebook
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

## Key Functions

### Preprocessing Functions
- `clean_text(subs)`: Preprocesses subtitle text
- `split_into_sentences(text)`: Segments text into sentences
- `tokenize_sentences(sentences)`: Tokenizes sentences into words

### Vectorization Functions
- `vectorize_sentences(tokenized_sentences)`: TF-IDF vectorization
- `vectorize_sentences2(tokenized_sentences)`: Doc2Vec vectorization
- `vectorize_sentences3(text)`: T5 model summarization
- `vectorize_sentences_4(sentences)`: SentenceTransformer embeddings
- `fast_text_vectorization(sentences)`: FastText embeddings

### Ranking Functions
- `rank_sentences(similarity_matrix)`: PageRank algorithm
- `rank_sentences_hits(similarity_matrix)`: HITS algorithm
- `create_similarity_matrix(tfidf_matrix)`: Cosine similarity computation

### Summarization Functions
- `tfidf_summary(cleaned_text, summary_size)`: TF-IDF based summarization
- `doc2vec_summary(cleaned_text, summary_size)`: Doc2Vec based summarization
- `transformer_summrizer(cleaned_text)`: SentenceTransformer based summarization
- `fast_text_summary(clean_text)`: FastText based summarization

## Future Improvements

- [ ] Implement additional evaluation metrics (BLEU, METEOR, BERTScore)
- [ ] Experiment with different summary lengths and compression ratios
- [ ] Fine-tune transformer models on domain-specific data
- [ ] Add support for multilingual summarization
- [ ] Implement hybrid extractive-abstractive approaches
- [ ] Optimize computational efficiency for large-scale processing
- [ ] Add real-time summarization pipeline
- [ ] Implement user study for human evaluation
- [ ] Create web interface for easy access
- [ ] Add visualization of sentence importance graphs

## Technical Details

### Similarity Matrix Construction
- All methods use cosine similarity between sentence vectors
- Creates fully connected graph where edge weights represent similarity
- Graph serves as input to ranking algorithms

### Summary Generation
- Selects top-k sentences based on ranking scores
- Maintains original sentence order in output
- Default summary size: 5 sentences

### Error Handling
- Implements try-except blocks for robust processing
- Filters corrupted T5 summaries (length < 100 characters)
- Handles RecursionError in ROUGE calculation

## Dependencies

- **NLTK**: Natural language processing and tokenization
- **Gensim**: Doc2Vec and FastText implementations
- **Scikit-learn**: TF-IDF vectorization and similarity metrics
- **NetworkX**: Graph algorithms (PageRank, HITS)
- **Transformers**: T5 model implementation
- **Sentence-Transformers**: Pre-trained sentence embeddings
- **ROUGE**: Evaluation metric computation
- **Pandas/NumPy**: Data manipulation and numerical operations

## References

- PageRank: Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual Web search engine
- HITS: Kleinberg, J. (1999). Authoritative sources in a hyperlinked environment
- Doc2Vec: Le, Q., & Mikolov, T. (2014). Distributed representations of sentences and documents
- T5: Raffel, C., et al. (2020). Exploring the limits of transfer learning
- SentenceTransformers: Reimers, N., & Gurevych, I. (2019). Sentence-BERT

## License

This project is available for educational and research purposes.

## Acknowledgments

- MrBeast YouTube channel for the dataset
- Hugging Face for pre-trained transformer models
- NLTK and Gensim communities for NLP tools
- Sentence-Transformers library developers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This project was developed as a comparative study of text summarization techniques. Results may vary based on dataset characteristics and hyperparameter tuning.
