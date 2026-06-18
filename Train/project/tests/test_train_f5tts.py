import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project.training.train_f5tts import F5TTSLauncher, LaunchConfig, main
from project.configs.defaults import ProjectConfig


class TrainF5TTSLauncherTests(unittest.TestCase):
    def test_main_uses_project_vocab_for_custom_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir) / "my_speak"
            project_dir.mkdir(parents=True, exist_ok=True)
            (project_dir / "raw.arrow").write_bytes(b"arrow")
            (project_dir / "duration.json").write_text("{}", encoding="utf-8")
            (project_dir / "vocab.txt").write_text("া\nক\n", encoding="utf-8")

            captured: dict[str, LaunchConfig] = {}

            def fake_launch(self: F5TTSLauncher, launch: LaunchConfig) -> int:
                captured["launch"] = launch
                return 0

            with patch.object(F5TTSLauncher, "recommend_settings", return_value={"batch_size_per_gpu": 4, "grad_accumulation_steps": 2, "num_workers": 4}), patch.object(F5TTSLauncher, "launch", fake_launch):
                exit_code = main([
                    "--project-name",
                    "my_speak",
                    "--project-dir",
                    str(project_dir),
                    "--dry-run",
                ])

            self.assertEqual(exit_code, 0)
            self.assertIn("launch", captured)
            self.assertEqual(captured["launch"].tokenizer_path, str(project_dir / "vocab.txt"))
            self.assertEqual(captured["launch"].project_dir, project_dir)


if __name__ == "__main__":
    unittest.main()
