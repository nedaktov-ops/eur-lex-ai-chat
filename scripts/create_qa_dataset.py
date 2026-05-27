#!/usr/bin/env python3
"""
create_qa_dataset.py — Generate ground-truth QA dataset for RAGAS evaluation.

Queries data/chunks.db for documents with good article coverage and creates
question-answer pairs from document titles and article headings.

Output format: data/qa_dataset.json (HuggingFace Dataset compatible)
Each entry: {"question": "...", "answer": "...", "contexts": [...], "reference": "CELEX", "articles": []}
"""

import json
import logging
import os
import sqlite3
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
# Use backup database since main chunks.db may be empty
DB_PATH = DATA_DIR / "backup-20260523-195951" / "chunks.db"
OUTPUT_PATH = DATA_DIR / "qa_dataset.json"

# Pattern for "Article X" followed by optional heading text
ARTICLE_HEADER_PATTERN = re.compile(r"Article\s+(\d+)\s*(?:[-–—]?\s*([^\n\r]{5,80})?)?", re.IGNORECASE)
# Pattern to detect if article is just a number with procedural text
PROCEDURAL_KEYWORDS = re.compile(r"\b(enter\s+into\s+force|shall\s+be\s+amended|shall\s+apply|shall\s+enter|repealed|published|official\s+journal)\b", re.IGNORECASE)
# Pattern to remove parenthetical abbreviations from titles
TITLE_ABBREV_PATTERN = re.compile(r"\b(\([A-Z]+\)|\([0-9]+\)|\([A-Z]+[0-9]+\))\b")


def clean_title_for_question(title):
    """Clean document title for use in questions."""
    # Remove parenthetical abbreviations (e.g., "(EC)", "2004")
    title = TITLE_ABBREV_PATTERN.sub("", title).strip()
    # Remove multiple spaces
    title = re.sub(r'\s+', ' ', title)
    return title


def extract_article_info_from_text(text):
    """Extract article number and heading/topic from article text."""
    # Look for "Article X" at the beginning
    match = ARTICLE_HEADER_PATTERN.search(text)
    if match:
        article_num = match.group(1)
        heading = match.group(2) if match.group(2) else None
        
        if heading:
            # Clean heading: remove trailing punctuation, truncate
            heading = heading.strip(" .,;:-")
            if len(heading) > 80:
                heading = heading[:80].strip() + "..."
            return article_num, heading
        else:
            # No explicit heading. Try to extract first complete sentence after "Article X"
            after_match = text[match.end():].strip()
            # Get first sentence (up to period, question mark, or exclamation)
            first_sentence_match = re.match(r"([^.!?]+[.!?])", after_match)
            if first_sentence_match:
                first_sentence = first_sentence_match.group(1).strip()
                if first_sentence:
                    # Truncate to reasonable length
                    if len(first_sentence) > 100:
                        first_sentence = first_sentence[:100].strip() + "..."
                    return article_num, first_sentence
            # Fallback: just use article number
            return article_num, None
    return None, None


def generate_question_from_article(title, article_num, heading=None):
    """Generate a question from an article heading."""
    if not article_num:
        return None
    
    clean_title = clean_title_for_question(title)
    
    # If we have a heading, use it to create a specific question
    if heading:
        heading_lower = heading.lower()
        # Remove "the" or "this regulation shall" etc. from beginning of heading
        heading_clean = re.sub(r"^(the\s+|this\s+regulation\s+shall\s+)", "", heading_lower).strip()
        
        if "principle" in heading_clean:
            return f"What are the {heading_clean} of {clean_title}?"
        elif "obligation" in heading_clean or "duty" in heading_clean:
            return f"What are the obligations regarding {heading_clean} in {clean_title}?"
        elif "right" in heading_clean:
            return f"What rights are established in {clean_title} regarding {heading_clean}?"
        elif "scope" in heading_clean or "application" in heading_clean:
            return f"What is the scope of {clean_title}?"
        elif "definition" in heading_clean or "meaning" in heading_clean:
            return f"What does {clean_title} define regarding {heading_clean}?"
        elif "purpose" in heading_clean:
            return f"What is the purpose of {clean_title}?"
        elif "requirement" in heading_clean:
            return f"What are the requirements of {clean_title}?"
        else:
            # Use a portion of the heading as the topic
            topic = heading_clean[:50].strip() if heading_clean else "its provisions"
            return f"What does Article {article_num} of {clean_title} establish regarding {topic}?"
    else:
        # No heading, just article number
        return f"What is Article {article_num} of {clean_title} about?"


def generate_question_from_title(title):
    """Generate a general question from document title."""
    clean_title = clean_title_for_question(title)
    return f"What does {clean_title} regulate?"


def load_chunks_from_db(db_path):
    """Load all chunks from the SQLite database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Load all chunks, ordered by celex and id
    cursor.execute("SELECT id, celex, title, article, type, text FROM chunks ORDER BY celex, id")
    rows = cursor.fetchall()
    conn.close()
    
    chunks = []
    for row in rows:
        chunks.append({
            "id": row["id"],
            "celex": row["celex"],
            "title": row["title"],
            "article": row["article"],
            "type": row["type"],
            "text": row["text"]
        })
    
    logger.info(f"Loaded {len(chunks)} chunks from database")
    return chunks


def group_chunks_by_document(chunks):
    """Group chunks by CELEX document."""
    by_doc = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk["celex"]].append(chunk)
    return by_doc


def select_documents_with_good_coverage(by_doc, min_articles=3, min_chunks=5):
    """Select documents that have sufficient article coverage."""
    selected = {}
    for celex, doc_chunks in by_doc.items():
        # Count unique articles (excluding None)
        articles = set(c["article"] for c in doc_chunks if c["article"])
        if len(articles) >= min_articles and len(doc_chunks) >= min_chunks:
            # Use the title from first chunk (should be consistent)
            title = doc_chunks[0]["title"]
            selected[celex] = {
                "title": title,
                "chunks": doc_chunks,
                "articles": list(articles)
            }
    logger.info(f"Selected {len(selected)} documents with good coverage (≥{min_articles} articles, ≥{min_chunks} chunks)")
    return selected


def create_qa_pairs(documents, target_count=100):
    """Create QA pairs from documents."""
    qa_pairs = []
    pair_id = 0
    
    for celex, doc_info in documents.items():
        title = doc_info["title"]
        chunks = doc_info["chunks"]
        articles = doc_info["articles"]
        
        # 1. Generate a general question from the title (1 per document)
        title_question = generate_question_from_title(title)
        if title_question:
            # Find chunks that contain the title or are introductory
            # Use first few chunks as context for the title question
            intro_chunks = [c["text"] for c in chunks[:3]]
            answer_text = " ".join(intro_chunks[:200])  # Truncate for answer
            
            # Convert article codes to readable form if possible
            readable_articles = []
            for art in articles:
                # Try to extract number from article codes like 'art_5'
                m = re.match(r"art_(\d+)", art)
                if m:
                    readable_articles.append(f"Article {m.group(1)}")
                else:
                    readable_articles.append(art)
            
            qa_pairs.append({
                "id": f"qa_{pair_id:04d}",
                "question": title_question,
                "answer": answer_text,
                "contexts": intro_chunks,
                "reference": "CELEX",
                "articles": readable_articles[:3],  # Include top 3 articles
                "celex": celex,
                "metadata": {"type": "title_based", "source": "document_title"}
            })
            pair_id += 1
        
        # 2. Generate questions from article headings (up to a few per document)
        article_chunks = [c for c in chunks if c["type"] == "section" and c["article"]]
        # Deduplicate by article text
        seen_articles = set()
        unique_article_chunks = []
        for c in article_chunks:
            if c["article"] not in seen_articles:
                seen_articles.add(c["article"])
                unique_article_chunks.append(c)
        
        # Pick up to 3 articles per document to generate specific questions
        for i, article_chunk in enumerate(unique_article_chunks[:3]):
            article_text = article_chunk["text"]
            article_code = article_chunk["article"]  # e.g., 'art_5'
            # Extract article number and heading from the text
            article_num, heading = extract_article_info_from_text(article_text)
            if not article_num:
                continue
            question = generate_question_from_article(title, article_num, heading)
            if question:
                # Find chunks that belong to this article
                article_chunks_context = [c["text"] for c in chunks if c["article"] == article_text]
                if not article_chunks_context:
                    article_chunks_context = [article_chunk["text"]]
                
                answer_text = " ".join(article_chunks_context[:2])  # Concatenate first 2 chunks
                
                qa_pairs.append({
                    "id": f"qa_{pair_id:04d}",
                    "question": question,
                    "answer": answer_text,
                    "contexts": article_chunks_context[:3],
                    "reference": "CELEX",
                    "articles": [f"Article {article_num}"],
                    "celex": celex,
                    "metadata": {"type": "article_based", "source": "article_heading"}
                })
                pair_id += 1
        
        # Stop if we've reached target count
        if len(qa_pairs) >= target_count:
            break
    
    logger.info(f"Generated {len(qa_pairs)} QA pairs")
    return qa_pairs


def save_dataset(qa_pairs, output_path):
    """Save QA pairs in HuggingFace Dataset compatible JSON format."""
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write as JSON lines (one dict per line) for HF Dataset compatibility
    with open(output_path, "w") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved dataset to {output_path} ({len(qa_pairs)} entries)")


def main():
    """Main pipeline."""
    logger.info("Starting QA dataset generation")
    
    # 1. Load chunks from database
    logger.info(f"Loading chunks from {DB_PATH}")
    chunks = load_chunks_from_db(DB_PATH)
    if not chunks:
        logger.error("No chunks found in database")
        return
    
    # 2. Group by document
    by_doc = group_chunks_by_document(chunks)
    
    # 3. Select documents with good coverage
    documents = select_documents_with_good_coverage(by_doc, min_articles=3, min_chunks=5)
    if not documents:
        logger.error("No documents meet coverage criteria")
        return
    
    # 4. Create QA pairs (target 100+)
    qa_pairs = create_qa_pairs(documents, target_count=120)  # Aim for 120 to ensure 100+
    
    # 5. Save dataset
    save_dataset(qa_pairs, OUTPUT_PATH)
    
    logger.info("Dataset generation complete")
    logger.info(f"Total documents used: {len(documents)}")
    logger.info(f"Total QA pairs: {len(qa_pairs)}")
    
    # Show a sample
    if qa_pairs:
        logger.info("\nSample entry:")
        sample = qa_pairs[0]
        print(f"  Question: {sample['question']}")
        print(f"  Answer: {sample['answer'][:100]}...")
        print(f"  Contexts: {len(sample['contexts'])} chunks")
        print(f"  Articles: {sample['articles']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Failed to generate dataset: {e}")
        exit(1)
