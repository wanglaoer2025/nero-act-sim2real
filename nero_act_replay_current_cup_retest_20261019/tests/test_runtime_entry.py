import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class RuntimeEntryTests(unittest.TestCase):
    def test_receipt_cannot_start_a_second_attempt_in_a_different_output(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.runtime'))
        from executor.runtime import consume_attempt
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory)/'receipt.json';receipt.write_text('{"confirmation":"ACT_FULL"}')
            with patch('executor.runtime.ATTEMPT_LEDGER', Path(directory)/'ledger', create=True):
                consume_attempt(receipt, Path(directory)/'first')
                with self.assertRaises(FileExistsError):consume_attempt(receipt, Path(directory)/'second')
            self.assertEqual(receipt.read_text(), '{"confirmation":"ACT_FULL"}')

    def test_copied_or_reformatted_receipt_cannot_authorize_another_attempt(self):
        from executor.runtime import consume_attempt
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root/'original.json'
            original.write_text('{"confirmation":"ACT_FULL", "authorized_wall_s":100}')
            duplicate = root/'copy.json'
            duplicate.write_text('{"authorized_wall_s":100,"confirmation":"ACT_FULL"}')
            with patch('executor.runtime.ATTEMPT_LEDGER', root/'ledger', create=True):
                marker = consume_attempt(original, root/'first')
                with self.assertRaises(FileExistsError):
                    consume_attempt(duplicate, root/'second')
                self.assertEqual(len(list((root/'ledger').glob('*.json'))), 1)
                self.assertEqual(json.loads(marker.read_text())['output'], str(root/'first'))

    def test_default_is_offline_and_execute_without_receipt_creates_no_attempt(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root/'run.py').exists())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory);model = path/'model';model.mkdir()
            (model/'policy.ckpt').write_bytes(b'weights');(model/'dataset_stats.pkl').write_bytes(b'stats')
            meta = dict(task_scope='complete_pick_carry_place_retreat', action_chunk_shape=[20, 8],
                        checkpoint_sha256=hashlib.sha256(b'weights').hexdigest(),
                        stats_sha256=hashlib.sha256(b'stats').hexdigest(), source_sha256={})
            (model/'runtime_metadata.json').write_text(json.dumps(meta))
            evaluation = path/'evaluation.json';evaluation.write_text(json.dumps(dict(passed=0,total=14)))
            cmd = [sys.executable, str(root/'run.py'), '--model', str(root/'model/frozen_run11'), '--evaluation', str(evaluation)]
            result = subprocess.run(cmd+['--output', str(path/'offline')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)['hardware_connected'])
            denied = subprocess.run(cmd+['--output', str(path/'denied'), '--execute'], capture_output=True, text=True)
            self.assertNotEqual(denied.returncode, 0)
            self.assertFalse((path/'denied').exists())


if __name__ == '__main__':unittest.main()
