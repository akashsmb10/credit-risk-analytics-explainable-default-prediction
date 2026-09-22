"""Load derived SQLite data and run the documented SQL statements."""
import sqlite3
from pathlib import Path

from data_loading import ROOT, load_data


def main():
    database = ROOT / "data" / "processed" / "credit_risk.sqlite"
    database.parent.mkdir(exist_ok=True)
    with sqlite3.connect(database) as connection:
        load_data().to_sql("credit_cards", connection, if_exists="replace", index=False)
        sql = (ROOT / "sql" / "credit_risk_analysis.sql").read_text(encoding="utf-8")
        # Remove full-line comments before separating complete SQL statements.
        sql_without_comments = "\n".join(
            line for line in sql.splitlines() if not line.strip().startswith("--")
        )
        statements = [statement.strip() for statement in sql_without_comments.split(";") if statement.strip()]
        for number, statement in enumerate(statements, start=1):
            print(f"Query {number}: {connection.execute(statement).fetchall()}")


if __name__ == "__main__":
    main()
