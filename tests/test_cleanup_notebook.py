"""Exercise the cleanup notebook against disposable checkpoint files only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

NOTEBOOK = Path(__file__).resolve().parents[1] / 'MicroDuck_Cleanup_Old_Checkpoints.ipynb'


class CleanupNotebookTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(NOTEBOOK.is_file(), 'Cleanup notebook is missing')
        self.notebook = json.loads(NOTEBOOK.read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.drive = Path(self.temp.name)
        self.root = self.drive / 'logs/rsl_rl/jump_spin'
        self.keep = self.root / 'final/model_9999.pt'
        self.old = self.root / 'early/model_100.pt'
        self.new = self.drive / 'logs/rsl_rl/jump_spin_launch_v2/run/model_100.pt'
        self.log = self.root / 'final/training.log'
        for path in (self.keep, self.old, self.new, self.log):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'preserve-test-content')

    def run_cleanup(self, delete=False, keep='final/model_9999.pt'):
        cell = next(c for c in self.notebook['cells'] if c['id'] == 'cleanup')
        namespace = {'DRIVE_ROOT': str(self.drive), 'KEEP_RELATIVE_PATH': keep, 'DELETE': delete}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(''.join(cell['source']), str(NOTEBOOK), 'exec'), namespace)

    def test_default_preview_leaves_all_files_untouched(self):
        config = next(c for c in self.notebook['cells'] if c['id'] == 'settings')
        namespace = {}
        exec(''.join(config['source']), namespace)
        self.assertIs(namespace['DELETE'], False)
        self.run_cleanup()
        self.assertTrue(self.old.exists())
        self.assertTrue(self.keep.exists())
        self.assertTrue(self.new.exists())

    def test_delete_only_old_checkpoints_preserving_final_and_new_experiment(self):
        self.run_cleanup(delete=True)
        self.assertFalse(self.old.exists())
        for path in (self.keep, self.new, self.log):
            self.assertEqual(path.read_bytes(), b'preserve-test-content')

    def test_missing_or_empty_final_aborts_before_deletion(self):
        self.keep.unlink()
        with self.assertRaises(RuntimeError):
            self.run_cleanup(delete=True)
        self.assertTrue(self.old.exists())
        self.keep.touch()
        with self.assertRaises(RuntimeError):
            self.run_cleanup(delete=True)
        self.assertTrue(self.old.exists())

    def test_final_outside_old_experiment_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_cleanup(delete=True, keep='../jump_spin_launch_v2/run/model_100.pt')
        self.assertTrue(self.old.exists())
        self.assertTrue(self.new.exists())


if __name__ == '__main__':
    unittest.main()
