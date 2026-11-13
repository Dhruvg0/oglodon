"""``python -m oglodon`` (also what ``torchrun -m oglodon train ...`` launches)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
