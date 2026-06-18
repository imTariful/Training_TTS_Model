import unittest

from project.preprocessing.quality_filter import QualityThresholds, filter_sample
from project.utils.metadata import MetadataRow


class QualityFilterTests(unittest.TestCase):
    def test_filters_low_quality_sample(self) -> None:
        row = MetadataRow(
            values={
                "transcription_confidence": "0.94",
                "audio_quality_score": "0.95",
                "speaker_verification_score": "0.95",
                "snr": "25",
                "cer": "0.01",
                "duration": "5",
            }
        )
        decision = filter_sample(row, QualityThresholds())
        self.assertFalse(decision.keep)
        self.assertIn("transcription_confidence", decision.reasons)


if __name__ == "__main__":
    unittest.main()
