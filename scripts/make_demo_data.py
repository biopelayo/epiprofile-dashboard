#!/usr/bin/env python3
"""Generate a small synthetic EpiProfile-Plants output directory.

The result mimics the real file layout closely enough for the dashboard to
load it: a ``histone_ratios.xls`` with Ratios / Areas / RT blocks separated by
empty columns, a ``histone_ratios_single_PTMs.xls`` and a phenodata TSV.
Values are random and carry no biological meaning.

Usage:
    python scripts/make_demo_data.py [output_dir] [--seed N]

Default output: examples/demo_experiment (relative to the repository root).
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (sequence, region, [peptidoforms])
REGIONS = [
    ("TKQTAR", "H3_3_8", ["unmod", "K4me1", "K4me2", "K4me3"]),
    ("KSTGGKAPR", "H3_9_17", ["unmod", "K9me1", "K9me2", "K9me3", "K14ac", "K9me2K14ac", "K9ac"]),
    ("KQLATKAAR", "H3_18_26", ["unmod", "K18ac", "K23ac", "K18acK23ac"]),
    ("KSAPATGGVKKPHR", "H3_27_40", ["unmod", "K27me1", "K27me2", "K27me3", "K36me1", "K36me2", "K36me3", "K27me1K36me2"]),
    ("EIAQDFKTDLR", "H3_73_83", ["unmod", "K79me1", "K79me2", "K79me3"]),
    ("GKGGKGLGKGGAKR", "H4_4_17", ["unmod", "K5ac", "K8ac", "K12ac", "K16ac", "K5acK8ac", "K8acK16ac", "K5acK8acK12acK16ac"]),
    ("KVLR", "H4_20_23", ["unmod", "K20me1", "K20me2", "K20me3"]),
]

SINGLE_PTMS = [
    "H3K4me1", "H3K4me2", "H3K4me3", "H3K9me1", "H3K9me2", "H3K9me3", "H3K9ac",
    "H3K14ac", "H3K18ac", "H3K23ac", "H3K27me1", "H3K27me2", "H3K27me3",
    "H3K36me1", "H3K36me2", "H3K36me3", "H3K79me1", "H3K79me2", "H3K79me3",
    "H4K5ac", "H4K8ac", "H4K12ac", "H4K16ac", "H4K20me1", "H4K20me2", "H4K20me3",
]

GROUPS = {"Control": "Shoot", "TSA": "Shoot", "NaB": "Root"}
REPLICATES = 4


def sample_names():
    names = []
    for g, tissue in GROUPS.items():
        for r in range(1, REPLICATES + 1):
            names.append(f"{g}_{tissue[:2]}{r}")
    return names


def build_ratios(rng, samples):
    n = len(samples)
    header = ["Peptide/PTM"] + [f"{i + 1},{s}" for i, s in enumerate(samples)]
    # Dedup-safe: pandas renames duplicate column names with .1/.2 suffixes,
    # which the dashboard strips, so we can repeat the sample header per block.
    line = header + [""] + header[1:] + [""] + header[1:]
    rows = ["\t".join(line)]
    # Sub-header row (skipped by the parser)
    sub = ["Ratio"] + ["ratio"] * n + [""] + ["area"] * n + [""] + ["RT"] * n
    rows.append("\t".join(sub))

    for seq, region, forms in REGIONS:
        # Region header row: sum of areas, no ratio values beyond total=1
        hdr_ratio = ["1"] * n
        hdr_area = [f"{v:.0f}" for v in rng.uniform(2e7, 8e7, n)]
        hdr_rt = [""] * n
        rows.append("\t".join([f"{seq}({region})"] + hdr_ratio + [""] + hdr_area + [""] + hdr_rt))

        # Dirichlet proportions per sample, with a group effect on the first mark
        alpha = np.ones(len(forms)) * 2.0
        props = rng.dirichlet(alpha, size=n).T  # forms x samples
        # Add a treatment effect: TSA raises acetylation, NaB raises methylation
        for j, s in enumerate(samples):
            for i, f in enumerate(forms):
                if s.startswith("TSA") and "ac" in f:
                    props[i, j] *= 1.8
                if s.startswith("NaB") and "me" in f:
                    props[i, j] *= 1.5
            props[:, j] /= props[:, j].sum()

        base_rt = rng.uniform(15, 60)
        for i, f in enumerate(forms):
            ratio = props[i]
            area = ratio * np.array([float(a) for a in hdr_area])
            # Random non-detects
            miss = rng.random(n) < 0.08
            ratio_s = ["0" if m else f"{v:.6f}" for v, m in zip(ratio, miss)]
            area_s = ["0" if m else f"{v:.0f}" for v, m in zip(area, miss)]
            rt_s = ["" if m else f"{base_rt + i * 0.7 + rng.normal(0, 0.15):.2f}" for m in miss]
            rows.append("\t".join([f"{region} {f}"] + ratio_s + [""] + area_s + [""] + rt_s))
    return "\n".join(rows) + "\n"


def build_single_ptms(rng, samples):
    n = len(samples)
    header = ["PTM"] + [f"{i + 1},{s}" for i, s in enumerate(samples)]
    rows = ["\t".join(header)]
    for ptm in SINGLE_PTMS:
        base = rng.uniform(0.02, 0.4)
        vals = np.clip(rng.normal(base, base * 0.25, n), 0, 1)
        for j, s in enumerate(samples):
            if s.startswith("TSA") and ptm.endswith("ac"):
                vals[j] *= 1.6
            if s.startswith("NaB") and "me" in ptm:
                vals[j] *= 1.4
        rows.append("\t".join([ptm] + [f"{v:.6f}" for v in vals]))
    return "\n".join(rows) + "\n"


def build_phenodata(samples):
    rows = ["Sample_Name\tSample_Group\tTissue\tDesign\tBatch"]
    for s in samples:
        group = s.split("_")[0]
        tissue = GROUPS[group]
        rows.append(f"{s}\t{group}\t{tissue}\tA\t1")
    return "\n".join(rows) + "\n"


def build_logs(samples):
    out = []
    for s in samples:
        out.append(s)
        out.append("..".join(r for _, r, _ in REGIONS))
        out.append("elapsed 12.3 s")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default=os.path.join(ROOT, "examples", "demo_experiment"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    samples = sample_names()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "histone_ratios.xls"), "w") as f:
        f.write(build_ratios(rng, samples))
    with open(os.path.join(a.out, "histone_ratios_single_PTMs.xls"), "w") as f:
        f.write(build_single_ptms(rng, samples))
    with open(os.path.join(a.out, "phenodata.tsv"), "w") as f:
        f.write(build_phenodata(samples))
    with open(os.path.join(a.out, "histone_logs.txt"), "w") as f:
        f.write(build_logs(samples))
    print(f"Demo experiment written to {a.out} ({len(samples)} samples, {len(REGIONS)} regions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
