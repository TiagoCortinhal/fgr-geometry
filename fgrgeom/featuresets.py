import json
import numpy as np
from fgrgeom import panel as P
from fgrgeom import config as C

# Named feature sets over the full loaded variable space. Each value is an
# include-tuple passed to panel.flatten. "full" deliberately excludes the sparse
# raw_doppler block (28s/32s PI/CPR ~3-20% observed) to avoid destroying
# complete-case coverage; access it explicitly via include=("raw_doppler",).
SETS = {
    "minimal":      ("biom", "doppler"),
    "plus_ratios":  ("biom", "ratios", "doppler"),
    "plus_cardiac": ("biom", "doppler", "cardiac"),
    "plus_maternal": ("biom", "doppler", "maternal"),
    "full":         ("biom", "ratios", "doppler", "cardiac", "maternal", "bp"),
}


def build(panel, name):
    if name not in SETS:
        raise KeyError(f"unknown feature set {name!r}; have {sorted(SETS)}")
    return P.flatten(panel, include=SETS[name])


def summary(panel, name):
    X, M, names = build(panel, name)
    return {
        "name": name,
        "include": list(SETS[name]),
        "n_fetuses": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "observed_fraction": float(M.mean()),
        "n_complete_case": int(M.all(axis=1).sum()),
    }


def main():
    panel = P.load_panel()
    table = [summary(panel, name) for name in SETS]
    out = {"n": int(len(panel.ids)), "sets": table}
    import pathlib
    res = pathlib.Path(__file__).resolve().parents[1] / "results"
    res.mkdir(exist_ok=True)
    with open(res / "featuresets.json", "w") as f:
        json.dump(out, f, indent=2)
    for r in table:
        print(f"{r['name']:>13}  feat={r['n_features']:>3}  "
              f"obs={r['observed_fraction']:.3f}  "
              f"complete={r['n_complete_case']:>4}/{r['n_fetuses']}")
    return out


if __name__ == "__main__":
    main()
