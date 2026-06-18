from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from .normalize_bn import normalize_bangla_text
from .phonemizer_bn import phonemize_bangla
from .quality_filter import QualityThresholds, filter_sample
from ..configs.defaults import ProjectConfig
from ..utils.audio import load_audio
from ..utils.metadata import MetadataRow, iter_metadata_csv, resolve_path, write_metadata_csv

LOGGER = logging.getLogger(__name__)
BATCH_SIZE = 10000


@dataclass(slots=True)
class PreparedRow:
    utterance_id: str
    audio_path: str
    text: str
    speaker_id: str
    split: str
    duration: float
    phoneme_sequence: str
    gender: str = ""
    age_group: str = ""
    dialect: str = ""
    emotion: str = ""


@dataclass(slots=True)
class PreparedProjectArtifacts:
    project_dir: Path
    metadata_csv: Path
    metadata_detailed_csv: Path
    raw_arrow: Path
    duration_json: Path
    vocab_txt: Path
    train_txt: Path
    valid_txt: Path


def _process_single_row(
    row: MetadataRow,
    thresholds: QualityThresholds,
    root_dir: Path,
    wav_dir: Path,
    use_phonemes: bool,
    normalize_text: bool,
) -> Optional[PreparedRow]:
    """Process a single row in isolation (for parallel processing)."""
    try:
        if not filter_sample(row, thresholds).keep:
            return None

        source_audio = resolve_path(row.get_str("audio_path") or row.get_str("audio"), root_dir)
        if source_audio is None or not source_audio.exists():
            return None

        text = row.get_str("normalized_text") or row.get_str("text")
        if normalize_text:
            text = normalize_bangla_text(text)

        phoneme_sequence = row.get_str("phoneme_sequence")
        if not phoneme_sequence and use_phonemes:
            phoneme_sequence = phonemize_bangla(text)

        utterance_id = row.get_str("utterance_id") or source_audio.stem
        target_audio = wav_dir / f"{utterance_id}.wav"
        if target_audio != source_audio:
            shutil.copy2(source_audio, target_audio)

        samples, sample_rate = load_audio(target_audio)
        duration = float(samples.shape[0] / sample_rate) if sample_rate else row.get_float("duration")

        return PreparedRow(
            utterance_id=utterance_id,
            audio_path=str(target_audio.resolve()),
            text=text,
            speaker_id=row.get_str("speaker_id", "unknown"),
            split=row.get_str("split", "train"),
            duration=duration,
            phoneme_sequence=phoneme_sequence,
            gender=row.get_str("gender"),
            age_group=row.get_str("age_group"),
            dialect=row.get_str("dialect"),
            emotion=row.get_str("emotion"),
        )
    except Exception as e:
        LOGGER.warning(f"Failed to process row: {e}")
        return None


class F5TTSDatasetPreparer:
    def __init__(
        self,
        config: ProjectConfig | None = None,
        use_phonemes: bool = True,
        normalize_text: bool = True,
        thresholds: QualityThresholds | None = None,
        num_workers: Optional[int] = None,
    ) -> None:
        self.config = config or ProjectConfig().resolve()
        self.use_phonemes = use_phonemes
        self.normalize_text = normalize_text
        self.num_workers = num_workers
        self.thresholds = thresholds or QualityThresholds(
            transcription_confidence=self.config.quality.transcription_confidence,
            audio_quality_score=self.config.quality.audio_quality_score,
            speaker_verification_score=self.config.quality.speaker_verification_score,
            snr=self.config.quality.snr,
            cer=self.config.quality.cer,
            min_duration=self.config.quality.min_duration,
            max_duration=self.config.quality.max_duration,
        )

    def prepare_project(self, metadata_file: str | Path, project_name: str, output_root: str | Path | None = None) -> PreparedProjectArtifacts:
        root = Path(output_root) if output_root is not None else self.config.dataset.root_dir
        project_dir = (root / project_name).resolve()
        wav_dir = project_dir / "wavs"
        wav_dir.mkdir(parents=True, exist_ok=True)

        # Stream rows and process in parallel
        prepared_rows_iterator = self._prepare_rows_streaming(metadata_file, wav_dir)
        
        # Write all artifacts incrementally
        self._write_metadata_csv_streaming(project_dir, prepared_rows_iterator)
        
        # We need to re-read to build splits, duration, and vocab (or use a temp file)
        # Let's store processed rows in a temp file and read back to avoid memory issues
        temp_metadata = project_dir / ".temp_metadata.csv"
        prepared_rows = []
        if temp_metadata.exists():
            with temp_metadata.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    prepared_rows.append(PreparedRow(
                        utterance_id=r["utterance_id"],
                        audio_path=r["audio_path"],
                        text=r["text"],
                        speaker_id=r["speaker_id"],
                        split=r["split"],
                        duration=float(r["duration"]),
                        phoneme_sequence=r["phoneme_sequence"],
                        gender=r["gender"],
                        age_group=r["age_group"],
                        dialect=r["dialect"],
                        emotion=r["emotion"],
                    ))
            temp_metadata.unlink()
        
        self._write_train_valid_splits(project_dir, prepared_rows)
        self._write_duration_json(project_dir, prepared_rows)
        self._write_vocab(project_dir, prepared_rows)
        self._write_raw_arrow(project_dir, prepared_rows)

        return PreparedProjectArtifacts(
            project_dir=project_dir,
            metadata_csv=project_dir / "metadata.csv",
            metadata_detailed_csv=project_dir / "metadata_detailed.csv",
            raw_arrow=project_dir / "raw.arrow",
            duration_json=project_dir / "duration.json",
            vocab_txt=project_dir / "vocab.txt",
            train_txt=project_dir / "train.txt",
            valid_txt=project_dir / "valid.txt",
        )

    def _prepare_rows_streaming(self, metadata_file: str | Path, wav_dir: Path) -> Iterator[PreparedRow]:
        """Process rows in parallel, streaming results."""
        rows = list(iter_metadata_csv(metadata_file))
        LOGGER.info(f"Loaded {len(rows)} rows from metadata. Starting parallel processing...")
        
        # Use ProcessPoolExecutor for CPU-bound tasks (audio loading, normalization)
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for row in rows:
                futures.append(executor.submit(
                    _process_single_row,
                    row,
                    self.thresholds,
                    self.config.dataset.root_dir,
                    wav_dir,
                    self.use_phonemes,
                    self.normalize_text,
                ))
            
            processed_count = 0
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    processed_count += 1
                    if processed_count % 1000 == 0:
                        LOGGER.info(f"Processed {processed_count} rows...")
                    yield result
            LOGGER.info(f"Completed processing {processed_count} valid rows.")

    def _write_metadata_csv_streaming(self, project_dir: Path, rows_iterator: Iterator[PreparedRow]) -> None:
        """Write metadata CSVs incrementally and save a temp copy for later use."""
        metadata_path = project_dir / "metadata.csv"
        detailed_path = project_dir / "metadata_detailed.csv"
        temp_path = project_dir / ".temp_metadata.csv"
        
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames_basic = ["audio_file", "text"]
        fieldnames_detailed = [
            "utterance_id", "audio_path", "text", "speaker_id", "split", "duration",
            "phoneme_sequence", "gender", "age_group", "dialect", "emotion"
        ]
        
        with metadata_path.open("w", encoding="utf-8", newline="") as f_basic, \
             detailed_path.open("w", encoding="utf-8", newline="") as f_detailed, \
             temp_path.open("w", encoding="utf-8", newline="") as f_temp:
            
            writer_basic = csv.DictWriter(f_basic, fieldnames=fieldnames_basic)
            writer_detailed = csv.DictWriter(f_detailed, fieldnames=fieldnames_detailed)
            writer_temp = csv.DictWriter(f_temp, fieldnames=fieldnames_detailed)
            
            writer_basic.writeheader()
            writer_detailed.writeheader()
            writer_temp.writeheader()
            
            for row in rows_iterator:
                basic_row = {"audio_file": row.audio_path, "text": row.text}
                detailed_row = {
                    "utterance_id": row.utterance_id,
                    "audio_path": row.audio_path,
                    "text": row.text,
                    "speaker_id": row.speaker_id,
                    "split": row.split,
                    "duration": f"{row.duration:.6f}",
                    "phoneme_sequence": row.phoneme_sequence,
                    "gender": row.gender,
                    "age_group": row.age_group,
                    "dialect": row.dialect,
                    "emotion": row.emotion,
                }
                writer_basic.writerow(basic_row)
                writer_detailed.writerow(detailed_row)
                writer_temp.writerow(detailed_row)

    def _write_train_valid_splits(self, project_dir: Path, rows: list[PreparedRow]) -> None:
        train_rows = [row for row in rows if row.split.lower() == "train"]
        valid_rows = [row for row in rows if row.split.lower() == "valid"]
        if not valid_rows and train_rows:
            split_index = max(1, int(len(train_rows) * (1.0 - self.config.dataset.valid_ratio)))
            valid_rows = train_rows[split_index:]
            train_rows = train_rows[:split_index]

        self._write_f5tts_split(project_dir / "train.txt", train_rows)
        self._write_f5tts_split(project_dir / "valid.txt", valid_rows)

    def _write_duration_json(self, project_dir: Path, rows: list[PreparedRow]) -> None:
        payload = {
            "sample_count": len(rows),
            "total_duration": sum(row.duration for row in rows),
        }
        (project_dir / "duration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_vocab(self, project_dir: Path, rows: list[PreparedRow]) -> None:
        vocab: set[str] = set()
        for row in rows:
            vocab.update(row.phoneme_sequence or row.text)
        (project_dir / "vocab.txt").write_text("\n".join(sorted(vocab)) + ("\n" if vocab else ""), encoding="utf-8")

    def _write_raw_arrow(self, project_dir: Path, rows: list[PreparedRow]) -> None:
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
        except Exception as exc:
            raise RuntimeError("pyarrow is required to write raw.arrow for F5-TTS training") from exc

        # Write in batches to reduce memory usage
        batch_size = BATCH_SIZE
        schema = pa.schema([
            ("audio_path", pa.string()),
            ("text", pa.string()),
            ("speaker_id", pa.string()),
            ("duration", pa.float64()),
            ("split", pa.string()),
            ("utterance_id", pa.string()),
            ("phoneme_sequence", pa.string()),
        ])
        
        with (project_dir / "raw.arrow").open("wb") as handle:
            with ipc.new_file(handle, schema) as writer:
                for i in range(0, len(rows), batch_size):
                    batch_rows = rows[i:i+batch_size]
                    table = pa.table(
                        {
                            "audio_path": [row.audio_path for row in batch_rows],
                            "text": [row.text for row in batch_rows],
                            "speaker_id": [row.speaker_id for row in batch_rows],
                            "duration": [row.duration for row in batch_rows],
                            "split": [row.split for row in batch_rows],
                            "utterance_id": [row.utterance_id for row in batch_rows],
                            "phoneme_sequence": [row.phoneme_sequence for row in batch_rows],
                        }
                    )
                    writer.write_table(table)
                    LOGGER.info(f"Wrote batch {i//batch_size + 1} to raw.arrow")

    def _write_f5tts_split(self, path: Path, rows: list[PreparedRow]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(f"{row.audio_path}|{row.text}|{row.speaker_id}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare an official F5-TTS project directory")
    parser.add_argument("--metadata-file", required=True)
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--skip-phonemes", action="store_true")
    parser.add_argument("--skip-normalization", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preparer = F5TTSDatasetPreparer(
        use_phonemes=not args.skip_phonemes, 
        normalize_text=not args.skip_normalization,
        num_workers=args.num_workers,
    )
    artifacts = preparer.prepare_project(args.metadata_file, args.project_name, output_root=args.output_root)
    LOGGER.info(f"Prepared F5-TTS project at {artifacts.project_dir}")
    print(str(artifacts.project_dir))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
