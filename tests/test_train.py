import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg

from train import train

ROOT = Path(__file__).resolve().parents[1]


class TrainingSmokeTest(unittest.TestCase):
    def test_import_and_training_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            # A fresh import must work without a CSV in the working directory,
            # and must not train or create output files.
            subprocess.run(
                [sys.executable, "-c",
                 f"import sys; sys.path.insert(0, {str(ROOT)!r}); import train"],
                cwd=tmp, check=True, capture_output=True, text=True,
            )
            self.assertEqual(list(output.iterdir()), [])

            scores = train(ROOT / "wine_quality.csv", output)
            self.assertEqual(set(scores), {"train_score", "test_score"})
            for score in scores.values():
                self.assertTrue(math.isfinite(score))
                self.assertLessEqual(score, 100)

            self.assertEqual(
                (output / "metrics.txt").read_text(),
                "Training variance explained: %2.1f%%\n"
                "Test variance explained: %2.1f%%\n"
                % (scores["train_score"], scores["test_score"]),
            )
            for name in ("feature_importance.png", "residuals.png"):
                self.assertGreater(mpimg.imread(output / name).size, 0)


if __name__ == "__main__":
    unittest.main()
