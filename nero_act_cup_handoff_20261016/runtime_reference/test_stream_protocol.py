import json
from pathlib import Path
import subprocess
import sys
import unittest


class StreamProtocolTests(unittest.TestCase):
    def test_native_sdk_stdout_cannot_corrupt_json_protocol(self):
        code = '''
import json, os
from stream_worker import isolate_protocol_stdout
protocol = isolate_protocol_stdout()
os.write(1, b'[ZED] native SDK log\\n')
print(json.dumps({'type': 'prediction', 'chunk': [1, 2]}), file=protocol, flush=True)
protocol.close()
'''
        result = subprocess.run([sys.executable, '-c', code],
                                cwd=Path(__file__).resolve().parent,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'type': 'prediction', 'chunk': [1, 2]})
        self.assertIn('[ZED] native SDK log', result.stderr)


if __name__ == '__main__':
    unittest.main()
