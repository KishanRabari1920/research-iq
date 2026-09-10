import sqlite3
import os


DATABASE = "data/papers.db"


def get_connection():

    # data folder create
    os.makedirs("data", exist_ok=True)

    return sqlite3.connect(DATABASE)



# CREATE PAPERS TABLE

def create_table():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            title TEXT NOT NULL,

            abstract TEXT,

            authors TEXT,

            date TEXT,

            link TEXT UNIQUE

        )
    """)

    connection.commit()

    connection.close()



# SAVE PAPERS

def save_papers(papers):

    connection = get_connection()

    cursor = connection.cursor()

    for paper in papers:

        cursor.execute("""
            INSERT OR IGNORE INTO papers
            (
                title,
                abstract,
                authors,
                date,
                link
            )

            VALUES (?, ?, ?, ?, ?)

        """, (

            paper["title"],

            paper["abstract"],

            ", ".join(paper["authors"]),

            paper["date"],

            paper["link"]

        ))

    connection.commit()

    connection.close()



# GET PAPERS

def get_papers():

    connection = get_connection()

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *

        FROM papers

        ORDER BY id DESC

    """)

    papers = cursor.fetchall()

    connection.close()

    return [
        dict(paper)
        for paper in papers
    ]