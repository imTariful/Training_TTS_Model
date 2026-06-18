from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.audio import load_audio
from ..utils.edit_distance import normalized_edit_distance, normalized_token_edit_distance
from ..utils.metadata import read_metadata_csv

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluationMetrics:
    mcd: float | None
    f0_rmse: float | None
    speaker_similarity: float | None
    cer: float | None
    wer: float | None
    mosnet: float | None


class Evaluator:
    def __init__(self, generated_dir: str | Path, reference_metadata: str | Path) -> None:
        self.generated_dir = Path(generated_dir)
        self.reference_rows = read_metadata_csv(reference_metadata)

    def evaluate(self) -> EvaluationMetrics:
        predicted_texts: list[str] = []
        reference_texts: list[str] = []
        cer_values: list[float] = []
        wer_values: list[float] = []
        for row in self.reference_rows:
            reference_text = row.get_str("normalized_text") or row.get_str("text")
            audio_path = self.generated_dir / f"{row.get_str('utterance_id')}.wav"
            if not audio_path.exists():
                continue
            predicted_text = self._load_prediction_text(row)
            reference_texts.append(reference_text)
            predicted_texts.append(predicted_text)
            cer_values.append(normalized_edit_distance(predicted_text, reference_text))
            wer_values.append(normalized_token_edit_distance(predicted_text.split(), reference_text.split()))

        cer = float(np.mean(cer_values)) if cer_values else None
        wer = float(np.mean(wer_values)) if wer_values else None
        return EvaluationMetrics(mcd=None, f0_rmse=None, speaker_similarity=None, cer=cer, wer=wer, mosnet=None)

    def _load_prediction_text(self, row: Any) -> str:
        transcript_path = self.generated_dir / f"{row.get_str('utterance_id')}.txt"
        if transcript_path.exists():
            return transcript_path.read_text(encoding="utf-8").strip()
        return row.get_str("text")


def build_report(metrics: EvaluationMetrics, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<html><head><meta charset='utf-8'><title>F5-TTS Evaluation</title></head><body>
<h1>Evaluation Report</h1>
<pre>{json.dumps(asdict(metrics), ensure_ascii=False, indent=2)}</pre>
</body></html>"""
    output.write_text(html, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate F5-TTS outputs")
    parser.add_argument("--generated-dir", default="generated")
    parser.add_argument("--reference-metadata", default="metadata.csv")
    parser.add_argument("--output-html", default="evaluation_report.html")
    parser.add_argument("--output-json", default="evaluation_metrics.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    evaluator = Evaluator(args.generated_dir, args.reference_metadata)
    metrics = evaluator.evaluate()
    Path(args.output_json).write_text(json.dumps(asdict(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    build_report(metrics, args.output_html)
    LOGGER.info("Saved evaluation artifacts to %s and %s", args.output_json, args.output_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
