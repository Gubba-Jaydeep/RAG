import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import argparse
import re

# Load embedding model and cross-encoder
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


# Load and preprocess data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['Year'] >= 2021]  # Filter latest two years
    df.fillna("Unknown", inplace=True)

    snippets = []
    for _, row in df.iterrows():
        snippets.append(
            f"In {row['Year']}, {row['Company']} ({row['Category']}) reported financials: "
            f"Market Cap: ${row['Market Cap(in B USD)']:.2f}B, Revenue: ${row['Revenue']:.2f}B, "
            f"Net Income: ${row['Net Income']:.2f}B, EPS: {row['Earning Per Share']:.2f}."
        )
        snippets.append(
            f"{row['Company']} had an EBITDA of ${row['EBITDA']:.2f}B with a Debt/Equity Ratio of {row['Debt/Equity Ratio']:.2f}. "
            f"ROE stood at {row['ROE']:.2f} while ROI was {row['ROI']:.2f}."
        )
        snippets.append(
            f"Financial health indicators for {row['Company']} in {row['Year']}: "
            f"Current Ratio: {row['Current Ratio']:.2f}, Free Cash Flow per Share: {row['Free Cash Flow per Share']:.2f}, "
            f"Number of Employees: {row['Number of Employees']}."
        )

    return df, snippets


# Indexing function
def build_index(snippets):
    tokenized_corpus = [doc.split() for doc in snippets]
    bm25 = BM25Okapi(tokenized_corpus)
    embeddings = embedding_model.encode(snippets, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return bm25, index, snippets, embeddings

# Guardrail: Input Validation
def validate_query(query):
    forbidden_patterns = [r'\bcapital of\b', r'\bwho is the president\b', r'\bweather\b']
    for pattern in forbidden_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False, "Invalid query: Please ask financial-related questions only."
    return True, ""

# Query function with Re-Ranking
def query_rag(query, bm25, index, snippets, embeddings, top_k=5):
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_k_bm25 = np.argsort(bm25_scores)[-top_k:][::-1]

    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    _, top_k_embed = index.search(query_embedding, top_k)

    retrieved_docs = list(set([snippets[i] for i in top_k_bm25] + [snippets[i] for i in top_k_embed[0]]))
    query_doc_pairs = [[query, doc] for doc in retrieved_docs]
    scores = cross_encoder.predict(query_doc_pairs)

    ranked_docs = sorted(zip(scores, retrieved_docs), reverse=True)
    best_doc = ranked_docs[0] if ranked_docs else (0, "No relevant information found.")

    # Output-Side Guardrail: Filter low-confidence responses
    confidence_threshold = 0.5  # Set threshold for relevance
    if best_doc[0] < confidence_threshold:
        return "Low confidence in response. Unable to provide an accurate answer."

    return f"Answer: {best_doc[1]}\nConfidence Score: {best_doc[0]:.2f}"


# CLI Interface
def main():
    exit_conditions = (":q", "quit", "exit")
    df, snippets = load_data('/Users/jaydeepgubba/Downloads/Financial Statements.csv')
    while True:
        query = input("Query> ")
        if query.lower().strip() in exit_conditions:
            print('\nByee!!')
            break
        is_valid, msg = validate_query(query)
        if not is_valid:
            print("\nResponse🪴 ", msg)
            continue
        bm25, index, snippets, embeddings = build_index(snippets)
        answer = query_rag(query, bm25, index, snippets, embeddings)
        print("\nResponse🪴 ", answer)


if __name__ == '__main__':
    main()
