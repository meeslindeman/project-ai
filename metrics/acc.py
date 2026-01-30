import re
import sys
from pathlib import Path
from statistics import mean, stdev

PATTERN = re.compile(r"corresponding test\s+([0-9]*\.?[0-9]+)")

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python summarize_tests.py /path/to/output_file.out", file=sys.stderr)
        sys.exit(2)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(2)

    text = path.read_text(encoding="utf-8", errors="replace")
    vals = [float(m.group(1)) for m in PATTERN.finditer(text)]

    if not vals:
        print("Error: no 'corresponding test <float>' entries found.", file=sys.stderr)
        sys.exit(1)

    m = mean(vals)
    s = stdev(vals) if len(vals) >= 2 else 0.0  # sample std

    print(f"Found {len(vals)} splits")
    print(f"Test mean: {m*100:.2f}%")
    print(f"Test sample std: {s*100:.2f}%")

if __name__ == "__main__":
    main()
