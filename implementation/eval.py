import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, _ResponseRelevancy

from answer import answer_eval


EVAL_FILE = Path(__file__).parent/'eval_set.jsonl'
OUTPUT_FILE = Path(__file__).parent/ "ragas_results.csv"


def load_eval_questions(path: Path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def build_ragas_rows(rows, limit=None):
    ragas_rows = []

    if limit:
        rows = rows[:limit]

    for i, row in enumerate(rows, start=1):
        question = row["user_input"]

        print(f"Running question {i}/{len(rows)}: {question}")

        answer, contexts = answer_eval(question)

        ragas_rows.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        })

    return ragas_rows


def main():
    rows = load_eval_questions(EVAL_FILE)

    
    ragas_rows = build_ragas_rows(rows, limit=20)

    dataset = Dataset.from_list(ragas_rows)

    result = evaluate(
        dataset,
        metrics=[faithfulness]
    )

    print(result)

    df = result.to_pandas()
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()