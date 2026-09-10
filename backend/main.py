from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from database import (
    create_table,
    save_papers,
    get_papers
)

from ingest import fetch_papers

from rag import (
    initialize_rag,
    build_indexes,
    search
)



# APP

app = FastAPI(
    title="Research RAG",
    version="1.0"
)



# CORS

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_methods=["*"],

    allow_headers=["*"]
)


# DATABASE
create_table()


# LOAD / BUILD RAG INDEX
initialize_rag()



# HOME
@app.get("/")
def home():

    return {
        "message": "Research RAG is running"
    }



# INGEST PAPERS
@app.post("/ingest")
async def ingest_papers(

    query: str = Query(
        default="biophysics",
        min_length=1,
        max_length=100,
        description="Search query for arXiv"
    ),

    total: int = Query(
        default=200,
        ge=1,
        le=200,
        description="Total number of papers"
    ),

    batch_size: int = Query(
        default=20,
        ge=1,
        le=20,
        description="Number of papers per API request"
    )

):


    # EXTRA VALIDATION
    if batch_size > total:

        return {
            "error":
                "batch_size cannot be greater than total",

            "total":
                total,

            "batch_size":
                batch_size
        }



    # SHOW PARAMETERS
    print(
        "\nStarting ingestion..."
    )

    print(
        "Query:",
        query
    )

    print(
        "Total:",
        total
    )

    print(
        "Batch size:",
        batch_size
    )



    # FETCH PAPERS
    all_papers = []


    for start in range(
        0,
        total,
        batch_size
    ):

        # Remaining papers
        remaining = total - start


        # Current batch
        current_batch_size = min(
            batch_size,
            remaining
        )

        print(
            f"\nFetching "
            f"{start} - "
            f"{start + current_batch_size}"
        )


        batch = await fetch_papers(

            search_query=query,

            start=start,

            max_results=current_batch_size

        )


        all_papers.extend(
            batch
        )


        
        # Stop if API returns no data
        if not batch:

            print(
                "No more papers found."
            )

            break



    # SAVE PAPERS
    save_papers(
        all_papers
    )



    # REBUILD INDEXES
    if all_papers:

        build_indexes()



    # RESPONSE
    return {

        "message":
            "Ingestion completed",

        "query":
            query,

        "total_requested":
            total,

        "batch_size":
            batch_size,

        "papers_downloaded":
            len(all_papers)

    }



# GET PAPERS
@app.get("/papers")
def get_all_papers():

    papers = get_papers()

    return {

        "total":
            len(papers),

        "papers":
            papers

    }



# ASK QUESTION
@app.get("/ask")
def ask_question(

    question: str

):

    if not question.strip():

        return {

            "error":
                "Question cannot be empty"

        }


    result = search(
        question
    )


    return {

        "question":
            question,

        "answer":
            result["answer"],

        "rewritten_query":
            result["rewritten_query"],

        "sources":
            result["sources"]

    }