#!/usr/bin/env python3
"""
classify_golden_dataset.py — Automated AI classification of Golden Dataset using Gemini 3.5 Flash Lite.

Populates the 'AI Labelling' column with 'sym' (symbolic) or 'sub' (substantive).
"""

import argparse
import asyncio
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from google import genai
from tqdm.asyncio import tqdm as async_tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert in Corporate Sustainability Reporting and ESG disclosure analysis.

Context:
Corporations face social expectations to demonstrate environmental and societal commitment to preserve legitimacy (Ashforth and Gibbs 1990; Suchman 1995). Firms manage this legitimacy through two approaches:

- SYMBOLIC ("sym"): Disclosure expressing general aspirations, values, policies, commitments, intentions, or positive claims concerning an SDG WITHOUT specific verifiable evidence of implementation, resource commitment, operational change, measurable progress, or achieved outcomes. Informational content is declarative or impression-oriented.
- SUBSTANTIVE ("sub"): Disclosure providing specific, potentially verifiable evidence of a firm's SDG-related implementation, resource commitment, operational change, measurable target, progress against baseline, or achieved outcome (e.g. quantified KPIs, allocated budgets, specific timelines, named projects with implementation evidence, third-party assurance).

Task:
Evaluate the Passage below with respect to the matched SDG Keyword pattern and classify the disclosure as either "sym" or "sub".

Output:
Return ONLY the exact label: "sym" or "sub". Do not include any extra text, punctuation, or explanations."""


def build_user_prompt(passage: str, keyword: str, sdg_category: str) -> str:
    return f"SDG Category: {sdg_category}\nMatched Keyword: {keyword}\n\nPassage:\n{passage}\n\nClassification (sym or sub):"


async def classify_one(client: genai.Client, passage: str, keyword: str, sdg_category: str, model_name: str, semaphore: asyncio.Semaphore) -> str:
    user_prompt = build_user_prompt(passage, keyword, sdg_category)
    async with semaphore:
        for attempt in range(5):
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model_name,
                    contents=user_prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                        max_output_tokens=10,
                    ),
                )
                text = response.text.strip().lower() if response.text else ""
                if "sub" in text:
                    return "sub"
                elif "sym" in text:
                    return "sym"
                return text[:10] if text else "sym"
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    wait_time = 4.0 * (attempt + 1)
                    await asyncio.sleep(wait_time)
                else:
                    await asyncio.sleep(1.0 * (attempt + 1))
        return "sym"


async def process_dataset(df: pd.DataFrame, api_key: str, model_name: str, concurrency: int) -> list[str]:
    client = genai.Client(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        classify_one(
            client=client,
            passage=str(row["passage"]),
            keyword=str(row["keyword"]),
            sdg_category=str(row.get("sdg_category", "")),
            model_name=model_name,
            semaphore=semaphore,
        )
        for _, row in df.iterrows()
    ]
    results = await async_tqdm.gather(*tasks, desc=f"Classifying with {model_name}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Classify golden dataset passages using Gemini")
    parser.add_argument("--key", default="AQ.Ab8RN6JJ0RS4eo9tx977YnqrgdChqwvMOcsfd6hejA7NIG8S9Q", help="Gemini API key")
    parser.add_argument("--model", default="gemini-3.5-flash-lite", help="Gemini model name")
    parser.add_argument("--input", default="data/golden_dataset_500.xlsx", help="Input golden dataset path")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent async requests")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        log.error(f"File not found: {in_path}")
        return

    df = pd.read_excel(in_path) if in_path.suffix == ".xlsx" else pd.read_csv(in_path)
    log.info(f"Loaded {len(df)} rows from {in_path}")
    log.info(f"Starting AI classification with Model={args.model}, Concurrency={args.concurrency}...")

    predictions = asyncio.run(process_dataset(df, args.key, args.model, args.concurrency))

    # Populate AI Labelling column
    df["AI Labelling"] = predictions
    if "chatgpt_prediction" in df.columns:
        df["chatgpt_prediction"] = predictions

    # Reorder columns to place AI Labelling prominently
    cols = df.columns.tolist()
    if "AI Labelling" in cols:
        cols.remove("AI Labelling")
        # Place AI Labelling next to human consensus
        insert_idx = cols.index("human_consensus") + 1 if "human_consensus" in cols else len(cols)
        cols.insert(insert_idx, "AI Labelling")
        df = df[cols]

    # Save to Excel & CSV
    df.to_excel(in_path, index=False, engine="openpyxl")
    csv_path = in_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    log.info(f"\nClassification complete! Saved to:")
    log.info(f"  - {in_path}")
    log.info(f"  - {csv_path}")

    # Summary
    counts = pd.Series(predictions).value_counts().to_dict()
    log.info(f"\nAI Labelling Distribution: {counts}")


if __name__ == "__main__":
    main()
