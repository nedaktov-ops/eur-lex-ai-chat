#!/usr/bin/env python3
"""
Benchmark index coverage against SPARQL total.

Steps:
1. Count distinct CELEXes in the index (from backup chunks.db)
2. Run SPARQL query to get total available documents (same criteria as build)
3. Calculate missing percentage
4. Year distribution of missing docs
5. Top 10 missing CELEXes by year (most recent)
6. Write markdown report
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import requests
from collections import defaultdict

# Configuration
PROJECT_ROOT = Path("/home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat")
DATA_DIR = PROJECT_ROOT / "data"
BACKUP_DB = DATA_DIR / "backup-20260523-195951" / "chunks.db"
SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"
DOC_TYPES = ["REG", "DIR"]
FROM_DATE = "2004-01-01"  # Same as build_index.py default

def get_indexed_celexes():
    """Get set of CELEXes present in the index."""
    conn = sqlite3.connect(str(BACKUP_DB))
    cursor = conn.execute("SELECT DISTINCT celex FROM chunks")
    celexes = {row[0] for row in cursor.fetchall()}
    conn.close()
    return celexes

def query_sparql_total():
    """Run SPARQL query to get all available documents (matching build criteria)."""
    type_filters_list = [
        f"?type = <http://publications.europa.eu/resource/authority/resource-type/{t}>"
        for t in DOC_TYPES
    ]
    type_filter = " ||\n    ".join(type_filters_list)

    query = f"""PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
SELECT DISTINCT ?celex ?date ?type
WHERE {{
    ?doc cdm:work_has_resource-type ?type .
    FILTER(
      {type_filter}
    )
    ?doc cdm:resource_legal_id_celex ?celex .
    OPTIONAL {{ ?doc cdm:work_date_document ?date . }}
    FILTER(?date >= "{FROM_DATE}T00:00:00"^^xsd:dateTime)
    FILTER(!CONTAINS(?celex, "R("))
}}
"""
    print(f"Querying SPARQL endpoint for total available documents...")
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; EUR-LexBenchmark/1.0; +https://github.com/your-repo)"
    }
    r = requests.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        timeout=120,
        headers=headers,
    )
    r.raise_for_status()
    data = r.json()
    bindings = data["results"]["bindings"]
    docs = []
    for b in bindings:
        docs.append({
            "celex": b["celex"]["value"],
            "date": b.get("date", {}).get("value", ""),
            "type": b["type"]["value"].split("/")[-1],
        })
    print(f"SPARQL returned {len(docs)} documents")
    return docs

def extract_year(date_str):
    """Extract year from date string (e.g., '2025-03-27' -> '2025')."""
    if not date_str:
        return None
    try:
        return date_str.split("-")[0]
    except:
        return None

def main():
    print("=== EUR-Lex Index Coverage Benchmark ===")
    
    # 1. Get indexed CELEXes
    print("\n1. Loading indexed CELEXes from backup...")
    indexed_celexes = get_indexed_celexes()
    indexed_count = len(indexed_celexes)
    print(f"   Indexed documents (distinct CELEXes): {indexed_count:,}")

    # 2. Query SPARQL for total available
    print("\n2. Querying SPARQL for total available documents...")
    all_docs = query_sparql_total()
    total_available = len(all_docs)
    print(f"   Total available documents from SPARQL: {total_available:,}")

    # 3. Calculate missing
    missing_count = total_available - indexed_count
    missing_percent = (missing_count / total_available * 100) if total_available > 0 else 0
    print("\n3. Coverage summary:")
    print(f"   Missing documents: {missing_count:,} ({missing_percent:.2f}%)")
    print(f"   Coverage: {indexed_count:,} / {total_available:,} = {(indexed_count/total_available*100):.2f}%")

    # 4. Year distribution of missing docs
    print("\n4. Computing year distribution of missing documents...")
    missing_by_year = defaultdict(int)
    missing_docs_details = []  # store for top 10
    
    for doc in all_docs:
        celex = doc["celex"]
        if celex not in indexed_celexes:
            year = extract_year(doc["date"])
            missing_by_year[year] += 1
            missing_docs_details.append(doc)
    
    # Sort years descending
    year_dist = sorted(missing_by_year.items(), key=lambda x: (x[0] if x[0] else ""), reverse=True)
    
    print("   Year distribution (missing):")
    for year, count in year_dist[:20]:  # show top 20 years
        if year:
            print(f"     {year}: {count:,}")
    
    # 5. Top 10 missing CELEXes by year (most recent first)
    print("\n5. Top 10 missing CELEXes (most recent):")
    # Sort by date descending
    missing_docs_details.sort(key=lambda x: x["date"], reverse=True)
    top10 = missing_docs_details[:10]
    for i, doc in enumerate(top10, 1):
        print(f"   {i:2}. {doc['celex']} ({doc['type']}, {doc['date']})")

    # 6. Write report
    today = datetime.now().strftime("%Y%m%d")
    report_path = PROJECT_ROOT / "docs" / f"coverage_benchmark_{today}.md"
    
    # Build markdown content
    md_lines = [
        "# EUR-Lex Index Coverage Benchmark",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Summary",
        f"- **Indexed documents:** {indexed_count:,}",
        f"- **Total available (SPARQL):** {total_available:,}",
        f"- **Missing documents:** {missing_count:,} ({missing_percent:.2f}%)",
        f"- **Coverage:** {(indexed_count/total_available*100):.2f}%",
        "",
        "## Year Distribution of Missing Documents",
        "| Year | Missing Count |",
        "|------|---------------|",
    ]
    for year, count in year_dist:
        year_display = year if year else "Unknown"
        md_lines.append(f"| {year_display} | {count:,} |")
    
    md_lines.extend([
        "",
        "## Top 10 Missing CELEXes (Most Recent)",
        "| # | CELEX | Type | Date |",
        "|---|-------|------|------|",
    ])
    for i, doc in enumerate(top10, 1):
        md_lines.append(f"| {i} | {doc['celex']} | {doc['type']} | {doc['date']} |")
    
    md_lines.extend([
        "",
        "## Methodology",
        "- **Index source:** `data/backup-20260523-195951/chunks.db`",
        "- **SPARQL query:** Filters: document types = REG, DIR; date >= 2004-01-01; excludes corrigenda (contains 'R(')",
        f"- **SPARQL endpoint:** {SPARQL_ENDPOINT}",
        "- Distinct CELEX count from SQLite: `SELECT COUNT(DISTINCT celex) FROM chunks`",
        "- Missing = SPARQL total - indexed distinct CELEXes",
        "- Year extracted from `work_date_document` field",
        "",
        "## Notes",
        "- The index is built from a backup as the current `chunks.db` is empty (0 bytes).",
        "- The backup was created on 2026-05-22 with 15,112 documents and 305,957 chunks.",
    ])
    
    content = "\n".join(md_lines)
    report_path.write_text(content)
    print(f"\n✓ Report written to: {report_path}")
    print(f"\nReturn: {report_path}")
    print(f"Key numbers: Indexed={indexed_count:,}, Total={total_available:,}, Missing={missing_count:,} ({missing_percent:.2f}%)")

if __name__ == "__main__":
    main()
