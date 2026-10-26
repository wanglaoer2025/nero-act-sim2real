"""Checked entry for future qualification; historical evaluator stays sealed."""
import qualify
from checked_runtime import load_projected


if __name__ == '__main__':
    qualify.load_projected = load_projected
    qualify.main()
