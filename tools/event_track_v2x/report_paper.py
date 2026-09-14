#!/usr/bin/env python3
"""Generate all four table files from independently evaluated, bound
records."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    from transvision.models.event_track_v2x.paper_reports import write_tables
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = write_tables(json.loads(Path(a.records).read_text()), a.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
