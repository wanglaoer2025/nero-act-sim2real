def test_qualification_entry_imports_without_initializing_a_policy():
    # In a fresh interpreter there is no earlier pipeline import to populate sys.path.
    import os
    from pathlib import Path
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c', 'import qualify'],
                            cwd=Path(__file__).resolve().parents[1],
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
