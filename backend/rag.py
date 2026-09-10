import os
import pickle
import requests

import faiss
import numpy as np

from database import get_papers
from rank_bm25 import BM25Okapi

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from langchain_text_splitters import RecursiveCharacterTextSplitter



# FILE PATHS

DATA_DIR = "data"

FAISS_FILE = os.path.join(
    DATA_DIR,
    "faiss.index"
)

BM25_FILE = os.path.join(
    DATA_DIR,
    "bm25.pkl"
)

CHUNKS_FILE = os.path.join(
    DATA_DIR,
    "chunks.pkl"
)



# GLOBAL VARIABLES

chunks = []

bm25 = None

faiss_index = None

embedding_model = None

reranker = None



# EMBEDDING MODEL

def load_embedding_model():

    global embedding_model

    if embedding_model is None:

        print("\nLoading embedding model...")

        embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Embedding model loaded.")



# CROSS ENCODER

def load_reranker():

    global reranker

    if reranker is None:

        print("\nLoading Cross Encoder...")

        reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        print("Cross Encoder loaded.")



# CREATE CHUNKS

def create_chunks():

    papers = get_papers()

    print(
        f"\nPapers loaded from SQLite: {len(papers)}"
    )


    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=150,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    new_chunks = []

    for paper in papers:

        title = paper.get("title", "") or ""

        abstract = paper.get("abstract", "") or ""

        text = (
            title
            + "\n\n"
            + abstract
        )

        splits = splitter.split_text(
            text
        )

        for chunk_text in splits:

            if not chunk_text.strip():
                continue

            new_chunks.append({

                "text": chunk_text,

                "title": title,

                "link": paper.get(
                    "link",
                    ""
                ),

                "authors": paper.get(
                    "authors",
                    ""
                ),

                "date": paper.get(
                    "date",
                    ""
                )

            })

    print(
        f"Chunks created: {len(new_chunks)}"
    )

    return new_chunks



# SAVE CHUNKS

def save_chunks():

    with open(
        CHUNKS_FILE,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    print(
        f"Chunks saved -> {CHUNKS_FILE}"
    )



# CREATE BM25

def create_bm25():

    global bm25

    print("\nCreating BM25 index...")

    tokenized_chunks = [

        chunk["text"]
        .lower()
        .split()

        for chunk in chunks

    ]

    bm25 = BM25Okapi(
        tokenized_chunks
    )

    with open(
        BM25_FILE,
        "wb"
    ) as file:

        pickle.dump(
            bm25,
            file
        )

    print(
        f"BM25 saved -> {BM25_FILE}"
    )



# CREATE FAISS

def create_faiss():

    global faiss_index

    print("\nCreating FAISS embeddings...")

    load_embedding_model()

    texts = [

        chunk["text"]

        for chunk in chunks

    ]

    embeddings = embedding_model.encode(

        texts,

        normalize_embeddings=True,

        show_progress_bar=True

    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(
        dimension
    )

    faiss_index.add(
        embeddings
    )

    faiss.write_index(

        faiss_index,

        FAISS_FILE

    )

    print(
        f"FAISS saved -> {FAISS_FILE}"
    )

    print(
        f"Total vectors: {faiss_index.ntotal}"
    )



# BUILD INDEXES

def build_indexes():

    global chunks

    print("\n")
    print("=" * 60)
    print("BUILDING RAG INDEXES")
    print("=" * 60)

    chunks = create_chunks()

    if not chunks:

        print(
            "No papers available. "
            "Cannot build indexes."
        )

        return

    save_chunks()

    create_bm25()

    create_faiss()

    print("\nIndexes created successfully.")

    print("=" * 60)



# LOAD INDEXES

def load_indexes():

    global chunks
    global bm25
    global faiss_index

    if not os.path.exists(
        CHUNKS_FILE
    ):

        return False

    if not os.path.exists(
        BM25_FILE
    ):

        return False

    if not os.path.exists(
        FAISS_FILE
    ):

        return False

    print("\n")
    print("=" * 60)
    print("LOADING SAVED RAG INDEXES")
    print("=" * 60)

    try:

        with open(
            CHUNKS_FILE,
            "rb"
        ) as file:

            chunks = pickle.load(
                file
            )

        with open(
            BM25_FILE,
            "rb"
        ) as file:

            bm25 = pickle.load(
                file
            )

        faiss_index = faiss.read_index(
            FAISS_FILE
        )

        print(
            f"Chunks loaded: {len(chunks)}"
        )

        print(
            f"FAISS vectors loaded: "
            f"{faiss_index.ntotal}"
        )

        print(
            "BM25 loaded successfully."
        )

        print("=" * 60)

        return True

    except Exception as error:

        print(
            "Error loading indexes:",
            error
        )

        return False



# INITIALIZE RAG

def initialize_rag():

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    loaded = load_indexes()

    if loaded:

        print(
            "\nUsing saved indexes."
        )

        return

    print(
        "\nNo saved indexes found."
    )

    build_indexes()



# QUERY REWRITING

def rewrite_query(question):

    prompt = f"""
Rewrite the following user question into a better
search query for finding relevant research papers.

Keep it close to the original meaning and keep all
important keywords/entities from the original question
(including proper nouns, product names, or company names
if present). Fix spelling mistakes. Do not introduce new
topics or terms that are not implied by the question.

Return ONLY the rewritten search query, nothing else.

Do not answer the question.

User question:
{question}

Search query:
"""

    try:

        response = requests.post(

            "http://localhost:11434/api/generate",

            json={

                "model": "llama3.2",

                "prompt": prompt,

                "stream": False

            },

            timeout=60

        )

        response.raise_for_status()

        data = response.json()

        rewritten = data.get(
            "response",
            ""
        ).strip()

        if not rewritten:

            return question

        return rewritten

    except Exception as error:

        print(
            "\nQuery rewriting failed:",
            error
        )

        print(
            "Using original question."
        )

        return question



# BM25 SEARCH

def bm25_search(
    query,
    k=8
):

    if bm25 is None:

        return []

    tokens = (
        query
        .lower()
        .split()
    )

    scores = bm25.get_scores(
        tokens
    )

    indexes = np.argsort(
        scores
    )[::-1]

    indexes = indexes[:k]

    return [
        int(index)
        for index in indexes
    ]



# VECTOR SEARCH

def vector_search(
    query,
    k=8
):

    if faiss_index is None:

        return []

    load_embedding_model()

    query_embedding = embedding_model.encode(

        [query],

        normalize_embeddings=True

    )

    query_embedding = np.asarray(

        query_embedding,

        dtype="float32"

    )

    

    actual_k = min(
        k,
        faiss_index.ntotal
    )

    if actual_k == 0:

        return []

    scores, indexes = faiss_index.search(

        query_embedding,

        actual_k

    )

    return [

        int(index)

        for index in indexes[0]

        if index >= 0

    ]



# RRF

def reciprocal_rank_fusion(

    keyword_results,

    vector_results

):

    scores = {}


    # BM25

    for rank, index in enumerate(
        keyword_results
    ):

        scores[index] = scores.get(
            index,
            0
        )

        scores[index] += (
            1 /
            (60 + rank + 1)
        )


    # Vector

    for rank, index in enumerate(
        vector_results
    ):

        scores[index] = scores.get(
            index,
            0
        )

        scores[index] += (
            1 /
            (60 + rank + 1)
        )


    # Sort

    ranked = sorted(

        scores.items(),

        key=lambda item: item[1],

        reverse=True

    )

    return [

        index

        for index, score in ranked

    ]



# CROSS ENCODER RERANK

def rerank(

    query,

    indexes,

    k=5

):

    if not indexes:

        return []

    load_reranker()

    candidates = []

    for index in indexes:

        if (
            index >= 0
            and
            index < len(chunks)
        ):

            candidates.append(
                chunks[index]
            )

    if not candidates:

        return []

    pairs = [

        [
            query,
            chunk["text"]
        ]

        for chunk in candidates

    ]

    scores = reranker.predict(
        pairs
    )

    results = list(
        zip(
            candidates,
            scores
        )
    )

    results.sort(

        key=lambda item: item[1],

        reverse=True

    )

    return [

        chunk

        for chunk, score
        in results[:k]

    ]



# OLLAMA

def ask_ollama(

    question,

    context

):

    prompt = f"""
You are a research assistant.

Answer the user's question based primarily
on the research context provided below.

The context comes from retrieved research papers.

If the context contains relevant information,
give a useful and clear answer.

Do NOT say that the answer is missing if the
context contains enough information.

If the context genuinely does not contain
enough information, then say:

"I could not find enough information in the papers."

Explain technical concepts in simple language.

RESEARCH CONTEXT:
-----------------

{context}

-----------------

USER QUESTION:
{question}

ANSWER:
"""

    try:

        response = requests.post(

            "http://localhost:11434/api/generate",

            json={

                "model":
                    "llama3.2",

                "prompt":
                    prompt,

                "stream":
                    False

            },

            timeout=120

        )

        response.raise_for_status()

        data = response.json()

        return data.get(
            "response",
            ""
        ).strip()

    except Exception as error:

        print(
            "\nOllama error:",
            error
        )

        return (
            "Ollama could not generate "
            "the answer."
        )



# COMPLETE RAG SEARCH

def search(question):

    print("\n")
    print("=" * 70)
    print("NEW QUESTION")
    print("=" * 70)

    print(
        f"\nOriginal Question:\n{question}"
    )

    
    # 1. QUERY REWRITE
    
    rewritten_query = rewrite_query(
        question
    )

    print(
        f"\nRewritten Query:\n{rewritten_query}"
    )


    
    # 2. BM25
    
    keyword_results = bm25_search(

        rewritten_query,

        k=8

    )

    print(
        "\nBM25 Results:"
    )

    print(
        keyword_results
    )


    
    # 3. VECTOR SEARCH
    
    vector_results = vector_search(

        rewritten_query,

        k=8

    )

    print(
        "\nVector Results:"
    )

    print(
        vector_results
    )


    
    # 4. RRF
    
    combined_results = reciprocal_rank_fusion(

        keyword_results,

        vector_results

    )

    print(
        "\nRRF Results:"
    )

    


    
    # 5. CROSS ENCODER
    
    final_chunks = rerank(

        rewritten_query,

        combined_results,

        k=5

    )

    

    for i, chunk in enumerate(
        final_chunks,
        start=1
    ):

        print(
            f"\n{i}. {chunk['title']}"
        )

        print(
            f"   {chunk['text'][:250]}..."
        )


    
    # 6. CONTEXT
    
    context_parts = []

    for i, chunk in enumerate(
        final_chunks,
        start=1
    ):

        context_parts.append(

            f"""
SOURCE {i}

TITLE:
{chunk["title"]}

AUTHORS:
{chunk["authors"]}

DATE:
{chunk["date"]}

LINK:
{chunk["link"]}

CONTENT:
{chunk["text"]}
"""

        )


    context = "\n\n".join(
        context_parts
    )


    print(
        "\nContext length:",
        len(context)
    )


    
    # 7. OLLAMA
    
    answer = ask_ollama(

        question,

        context

    )


    
    # 8. PRINT ANSWER IN TERMINAL
    
    print("\n")
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(
        f"\n{answer}"
    )


    
    # 9. SOURCES
    
    sources = [

        {
            "title":
                chunk["title"],

            "link":
                chunk["link"],

            # "authors":
            #     chunk["authors"],

        }

        for chunk in final_chunks

    ]


    
    # 10. RETURN TO API
    
    return {

        "answer":
            answer,

        "rewritten_query":
            rewritten_query,

        "sources":
            sources

    }
