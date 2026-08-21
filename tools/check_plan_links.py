"""Check that no two entries in plan_engine.LINK_SPECS are the same mechanism.

A duplicated link is not a harmless alias: discover_plans() treats each entry as its own
plan, so one reading competes as two, both can qualify, and both credit the same numbers
in the blended score -- a single mechanism counted twice in the final picks. That is
exactly what 'turning' and the traditional turning chart did before DUPLICATE_CHARTS.

Every link is probed on all 90 possible source numbers, so this compares the FUNCTIONS,
not their behaviour on one history.

    python tools/check_plan_links.py
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lottery_core import plan_engine as pl  # noqa: E402


# Two unrelated machine-number mappings. One is not enough: with mach = win + 1 the
# mach_carry link is indistinguishable from plus1 on every probe, which is a property of
# the probe, not of the links.
MACH_MAPS = (lambda x: (x * 37 + 11) % 90 + 1, lambda x: (x * 53 + 4) % 90 + 1)


def signature(fn):
    """How this link maps every single source number, plus a few multi-number draws so
    links that combine numbers (sum_pair, diff_pair, sum_triple) are distinguished too."""
    wins = [[x, x, x, x, x] for x in range(1, 91)]
    wins += [[1, 2, 3, 4, 5], [7, 19, 33, 56, 88], [10, 20, 30, 40, 50]]
    out = []
    for win in wins:
        for mach_map in MACH_MAPS:
            d = {'win': win, 'mach': [mach_map(x) for x in win], 'date': None, 'code': 'MS'}
            out.append(frozenset(fn(d)))
    return tuple(out)


def main():
    by_sig = defaultdict(list)
    for name, fn in pl.LINK_SPECS.items():
        by_sig[signature(fn)].append(name)
    collisions = sorted(v for v in by_sig.values() if len(v) > 1)
    print(f"{len(pl.LINK_SPECS)} links registered, {len(by_sig)} distinct mechanisms")
    if not collisions:
        print("PASS: every link is its own mechanism")
        return 0
    for group in collisions:
        print(f"  FAIL: {group} are the same function")
    print("\nRegister one entry per mechanism -- see plan_engine.DUPLICATE_CHARTS.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
