"""Launch v2 response probe; default remains entirely offline."""
import os
import sys
from pathlib import Path
root = Path(__file__).resolve().parent
arguments = sys.argv[1:]
for index, value in enumerate(arguments):
    if value == '--output' and index + 1 < len(arguments):
        arguments[index + 1] = str(Path(arguments[index + 1]).resolve())
    elif value.startswith('--output='):
        arguments[index] = '--output=' + str(Path(value.split('=', 1)[1]).resolve())
os.chdir(root / 'nero_act_source')
os.execv(sys.executable, [sys.executable, '-m', 'src.act.response_probe', *arguments])
