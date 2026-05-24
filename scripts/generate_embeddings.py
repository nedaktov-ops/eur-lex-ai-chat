#!/usr/bin/env python3
"""Generate FAISS embeddings from document chunks using ONNX BERT with sharding."""

import argparse
import json
import logging
import time

import numpy as np
import onnxruntime
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate FAISS embeddings from document chunks using ONNX BERT"
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to ONNX quantized model"
    )
    parser.add_argument(
        "--tokenizer-name",
        default="nlpaueb/bert-base-uncased-eurlex",
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--chunks", required=True, help="Path to JSON file with chunk data"
    )
    parser.add_argument(
        "--shard", type=int, default=0, help="Shard index for this process"
    )
    parser.add_argument(
        "--total-shards", type=int, default=1, help="Total number of shards"
    )
    parser.add_argument(
        "--output-dir", default="data/embeddings", help="Output directory"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max-length", type=int, default=512, help="Maximum token length"
    )
    return parser.parse_args(argv)


def mean_pooling(token_embeds, attention_mask):
    mask = attention_mask.astype(np.float32)
    mask_expanded = np.expand_dims(mask, axis=-1)
    mask_sum = np.sum(mask_expanded, axis=1)
    mask_sum = np.maximum(mask_sum, 1e-9)
    return np.sum(token_embeds * mask_expanded, axis=1) / mask_sum


def l2_normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return embeddings / norms


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Loading ONNX model from %s", args.model_path)
    session = onnxruntime.InferenceSession(
        args.model_path, providers=["CPUExecutionProvider"]
    )
    input_names = [inp.name for inp in session.get_inputs()]
    logger.info("Model loaded. Input names: %s", input_names)
    logger.info("Loading tokenizer: %s", args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    logger.info("Loading chunks from %s", args.chunks)
    with open(args.chunks) as f:
        chunks = json.load(f)

    shard = args.shard
    total_shards = args.total_shards
    my_chunks = chunks[shard::total_shards]
    logger.info(
        "Shard %d/%d: %d chunks out of %d total",
        shard,
        total_shards,
        len(my_chunks),
        len(chunks),
    )

    embed_dim = 768
    all_embeddings = np.empty((0, embed_dim), dtype=np.float32)
    celex_ids = []
    chunk_indices = []

    if len(my_chunks) == 0:
        logger.warning("Empty shard — writing empty arrays")
    else:
        for batch_start in range(0, len(my_chunks), args.batch_size):
            batch = my_chunks[batch_start : batch_start + args.batch_size]
            texts = [chunk["text"] for chunk in batch]

            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="np",
            )

            input_feed = {name: encoded[name].astype(np.int64) for name in input_names}

            outputs = session.run(None, input_feed)
            last_hidden = outputs[0]

            attention_mask = encoded.get(
                "attention_mask",
                np.ones((last_hidden.shape[0], last_hidden.shape[1]), dtype=np.int64),
            )

            pooled = mean_pooling(last_hidden, attention_mask)
            normalized = l2_normalize(pooled)
            all_embeddings = np.vstack([all_embeddings, normalized])

            for chunk in batch:
                celex_ids.append(chunk["celex"])
                chunk_indices.append(chunk.get("article", ""))

            if (batch_start // args.batch_size) % 50 == 0 and batch_start > 0:
                done = min(batch_start + args.batch_size, len(my_chunks))
                logger.info(
                    "Processed %d / %d chunks (shard %d)",
                    done,
                    len(my_chunks),
                    shard,
                )

    np.save(
        f"{args.output_dir}/embeddings_shard_{shard}.npy",
        all_embeddings.astype(np.float32),
    )

    metadata = {
        "shard": shard,
        "total_shards": total_shards,
        "count": len(all_embeddings),
        "celex_ids": celex_ids,
        "chunk_indices": chunk_indices,
        "model_name": args.model_path,
        "embed_dim": embed_dim,
    }

    import os

    os.makedirs(args.output_dir, exist_ok=True)

    with open(f"{args.output_dir}/metadata_shard_{shard}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - getattr(main, "_start_time", time.time())
    logger.info(
        "Shard %d complete: %d embeddings, dim=%d, shape=%s, time=%.2fs",
        shard,
        len(all_embeddings),
        embed_dim,
        all_embeddings.shape,
        elapsed,
    )


if __name__ == "__main__":
    main._start_time = time.time()
    main()
