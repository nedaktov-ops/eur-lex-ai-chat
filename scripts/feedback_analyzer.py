#!/usr/bin/env python3
"""Analyze pipeline logs for improvement opportunities.

Reads structured JSON pipeline logs (from Phase 0 logging middleware)
and produces actionable insights about system performance, failure patterns,
and improvement opportunities.

Usage:
    cat /path/to/server.log | python3 scripts/feedback_analyzer.py
    python3 scripts/feedback_analyzer.py --file /path/to/server.log
    python3 scripts/feedback_analyzer.py --file /path/to/server.log --days 7
"""
import json
import sys
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta


def parse_logs(lines):
    """Parse structured JSON log lines, yield (entry, line)."""
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            entry = json.loads(line)
            yield entry, line
        except json.JSONDecodeError:
            continue


def analyze(lines, days=None):
    """Analyze pipeline logs and return stats dict."""
    stats = {
        "total_queries": 0,
        "total_llm_calls": 0,
        "total_validation_passes": 0,
        "total_validation_failures": 0,
        "total_fallbacks": 0,
        "validation_failure_reasons": Counter(),
        "intent_breakdown": Counter(),
        "avg_latency_ms": 0,
        "latencies": [],
        "confidence_breakdown": Counter(),
        "answer_lengths": [],
        "citation_counts": [],
        "failed_queries": [],
    }

    sessions = {}  # request_id -> {stage: data}

    for entry, _ in parse_logs(lines):
        rid = entry.get("request_id")
        if not rid:
            continue
        if rid not in sessions:
            sessions[rid] = {}
        sessions[rid][entry.get("stage")] = entry.get("data", {})

    for rid, stages in sessions.items():
        # Count types
        if "answer_generated" in stages:
            stats["total_queries"] += 1
            ag = stages["answer_generated"]
            if ag.get("validation_passed") is True:
                stats["total_validation_passes"] += 1
                conf = ag.get("confidence_level", "unknown")
                stats["confidence_breakdown"][conf] += 1
            elif ag.get("validation_passed") is False:
                stats["total_validation_failures"] += 1
                stats["total_fallbacks"] += 1

        if "llm_call" in stages:
            stats["total_llm_calls"] += 1
            llm = stages["llm_call"]
            if llm.get("duration_ms"):
                stats["latencies"].append(llm["duration_ms"])

        if "query_processed" in stages:
            qp = stages["query_processed"]
            cl = qp.get("classification", {}) or {}
            intent = cl.get("legal_intent", "unknown")
            stats["intent_breakdown"][intent] += 1

        if "answer_generated" in stages:
            ag = stages["answer_generated"]
            al = ag.get("answer_length", 0)
            if al > 0:
                stats["answer_lengths"].append(al)
            cc = ag.get("citations_count", 0)
            stats["citation_counts"].append(cc)

    if stats["latencies"]:
        stats["avg_latency_ms"] = sum(stats["latencies"]) / len(stats["latencies"])
        stats["p95_latency_ms"] = sorted(stats["latencies"])[
            int(len(stats["latencies"]) * 0.95)
        ] if len(stats["latencies"]) >= 20 else max(stats["latencies"])
        stats["p50_latency_ms"] = sorted(stats["latencies"])[
            len(stats["latencies"]) // 2
        ]
    else:
        stats["avg_latency_ms"] = 0
        stats["p95_latency_ms"] = 0
        stats["p50_latency_ms"] = 0

    if stats["answer_lengths"]:
        stats["avg_answer_length"] = sum(stats["answer_lengths"]) / len(stats["answer_lengths"])
    else:
        stats["avg_answer_length"] = 0

    if stats["citation_counts"]:
        stats["avg_citations"] = sum(stats["citation_counts"]) / len(stats["citation_counts"])
    else:
        stats["avg_citations"] = 0

    return stats


def generate_insights(stats):
    """Generate human-readable insights from stats."""
    insights = []

    total = stats["total_queries"]
    if total == 0:
        return ["No queries found in logs."]

    pass_rate = (stats["total_validation_passes"] / total * 100) if total else 0
    fallback_rate = (stats["total_fallbacks"] / total * 100) if total else 0

    insights.append(f"📊 Total queries analyzed: {total}")
    insights.append(f"✅ Validation pass rate: {pass_rate:.1f}% ({stats['total_validation_passes']}/{total})")
    insights.append(f"❌ Fallback rate: {fallback_rate:.1f}% ({stats['total_fallbacks']}/{total})")
    insights.append(f"⏱  Avg latency: {stats['avg_latency_ms']:.0f}ms (p50: {stats['p50_latency_ms']:.0f}ms, p95: {stats['p95_latency_ms']:.0f}ms)")
    insights.append(f"📝 Avg answer length: {stats['avg_answer_length']:.0f} chars")
    insights.append(f"📎 Avg citations per answer: {stats['avg_citations']:.1f}")

    if stats["intent_breakdown"]:
        insights.append(f"\n📋 Query intent breakdown:")
        for intent, count in stats["intent_breakdown"].most_common():
            pct = count / total * 100
            insights.append(f"  {intent}: {count} ({pct:.0f}%)")

    if stats["confidence_breakdown"]:
        insights.append(f"\n📈 Confidence distribution:")
        total_conf = sum(stats["confidence_breakdown"].values())
        for level, count in stats["confidence_breakdown"].most_common():
            pct = count / total_conf * 100
            insights.append(f"  {level}: {count} ({pct:.0f}%)")

    return insights


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze EUR-Lex AI Chat pipeline logs")
    parser.add_argument("--file", help="Path to server log file (reads stdin if not set)")
    parser.add_argument("--days", type=int, help="Filter to last N days")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of insights")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    stats = analyze(lines, days=args.days)

    if args.json:
        print(json.dumps(stats, indent=2, default=str))
    else:
        insights = generate_insights(stats)
        print("\n".join(insights))


if __name__ == "__main__":
    main()
