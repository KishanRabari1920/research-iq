import httpx
import feedparser

from database import save_papers


ARXIV_URL = "https://export.arxiv.org/api/query"


async def fetch_papers(
    search_query="all:biophysics",
    start=0,
    max_results=20
):

    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results
    }

    async with httpx.AsyncClient(timeout=30) as client:

        response = await client.get(
            ARXIV_URL,
            params=params
        )

        response.raise_for_status()

    feed = feedparser.parse(response.content)

    papers = []

    for entry in feed.entries:

        paper = {
            "title": entry.title.strip(),

            "abstract": entry.summary.strip(),

            "authors": [
                author.name
                for author in entry.authors
            ],

            "date": entry.published,

            "link": entry.id
        }

        papers.append(paper)

    return papers


async def ingest():

    papers = await fetch_papers(
        search_query="all:biophysics",
        max_results=20
    )

    save_papers(papers)

    return len(papers)