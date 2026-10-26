"""The identical guarded evaluator; only the fixed-last candidate directory changes."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'nero_act_dense_response_20261023'))
import evaluate_dense

if __name__ == '__main__':
    evaluate_dense.ROOT = ROOT
    evaluate_dense.main()
