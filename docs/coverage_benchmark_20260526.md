# EUR-Lex Index Coverage Benchmark
**Generated:** 2026-05-26T02:14:24.115099

## Summary
- **Indexed documents:** 15,112
- **Total available (SPARQL):** 17,417
- **Missing documents:** 2,305 (13.23%)
- **Coverage:** 86.77%

## Year Distribution of Missing Documents
| Year | Missing Count |
|------|---------------|
| 2026 | 69 |
| 2025 | 186 |
| 2024 | 300 |
| 2023 | 108 |
| 2019 | 6 |
| 2018 | 1 |
| 2016 | 113 |
| 2015 | 321 |
| 2014 | 201 |
| 2013 | 31 |
| 2012 | 1 |
| 2011 | 1 |
| 2010 | 15 |
| 2009 | 2 |
| 2008 | 3 |
| 2007 | 2 |
| 2006 | 31 |
| 2005 | 1 |
| 2004 | 913 |

## Top 10 Missing CELEXes (Most Recent)
| # | CELEX | Type | Date |
|---|-------|------|------|
| 1 | 32026R1139 | REG | 2026-05-20 |
| 2 | 32026R1047 | REG | 2026-04-29 |
| 3 | 32026R1030 | REG | 2026-04-29 |
| 4 | 32026R0995 | REG | 2026-04-29 |
| 5 | 32026R1046 | REG | 2026-04-29 |
| 6 | 32026L1024 | DIR | 2026-04-29 |
| 7 | 32026L1021 | DIR | 2026-04-29 |
| 8 | 32026R0909 | REG | 2026-04-27 |
| 9 | 32026R0469 | REG | 2026-04-23 |
| 10 | 32026R0511 | REG | 2026-04-23 |

## Methodology
- **Index source:** `data/backup-20260523-195951/chunks.db`
- **SPARQL query:** Filters: document types = REG, DIR; date >= 2004-01-01; excludes corrigenda (contains 'R(')
- **SPARQL endpoint:** https://publications.europa.eu/webapi/rdf/sparql
- Distinct CELEX count from SQLite: `SELECT COUNT(DISTINCT celex) FROM chunks`
- Missing = SPARQL total - indexed distinct CELEXes
- Year extracted from `work_date_document` field

## Notes
- The index is built from a backup as the current `chunks.db` is empty (0 bytes).
- The backup was created on 2026-05-22 with 15,112 documents and 305,957 chunks.