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
import pandas as pd

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


import base64
import struct


def build_sample_folders(rng, out, samples):
    """histone_layouts/<i>_<sample>/detail/{H3_07_73_83.xls, H3_03_9_17.xls, psm/identification_list.xls}"""
    layouts = os.path.join(out, "histone_layouts")
    regions = {"H3_07_73_83.xls": ["K79me1", "K79me2", "K79me3", "unmod"],
               "H3_03_9_17.xls": ["K9me1", "K9me2", "K9me3", "K14ac", "K9me2K14ac", "unmod"]}
    peptides = {"EIAQDFKTDLR": "H3_73_83", "KSTGGKAPR": "H3_9_17", "KQLATKAAR": "H3_18_26",
                "GKGGKGLGKGGAKR": "H4_4_17", "TKQTAR": "H3_3_8"}
    for i, s in enumerate(samples, start=1):
        ddir = os.path.join(layouts, f"{i}_{s}", "detail")
        os.makedirs(os.path.join(ddir, "psm"), exist_ok=True)
        for fn, mods in regions.items():
            lines = []
            for block in ("ratio", "area", "rt"):
                lines.append(f"[{block}]")
                lines.append("mod\tz2\tz3\tz4")
                for m in mods:
                    if block == "ratio":
                        v = rng.dirichlet(np.ones(3)) * rng.uniform(0.05, 0.6)
                    elif block == "area":
                        v = rng.uniform(1e5, 5e7, 3) * np.array([1.0, 0.4, 0.1])
                    else:
                        base = 30 + 3 * mods.index(m)
                        v = base + rng.normal(0, 0.1, 3)
                    lines.append("\t".join([m] + [f"{x:.4f}" for x in v]))
                lines.append("")
            with open(os.path.join(ddir, fn), "w") as f:
                f.write("\n".join(lines))
        rows = ["Scan\tPeptide\tCharge\tXCorr\tRT\tObserved_mz\tModifications"]
        n_psm = int(rng.integers(60, 140))
        for k in range(n_psm):
            pep = list(peptides)[int(rng.integers(0, len(peptides)))]
            z = int(rng.choice([2, 3, 4], p=[0.6, 0.3, 0.1]))
            rows.append(f"{1000 + k * 7}\t{pep}\t{z}\t{rng.uniform(1.2, 5.5):.3f}\t{rng.uniform(10, 70):.2f}\t"
                        f"{rng.uniform(400, 900):.4f}\t{'K79me2' if pep.startswith('EIAQ') else ''}")
        with open(os.path.join(ddir, "psm", "identification_list.xls"), "w") as f:
            f.write("\n".join(rows) + "\n")


def build_dinosaur(rng, out, samples):
    """Fake Dinosaur *.features.tsv for the first two runs (schema of Dinosaur 1.2)."""
    ddir = os.path.join(out, "dinosaur")
    os.makedirs(ddir, exist_ok=True)
    cols = ["mz", "mostAbundantMz", "charge", "rtStart", "rtApex", "rtEnd", "fwhm", "nIsotopes", "nScans",
            "averagineCorr", "mass", "massCalib", "intensityApex", "intensitySum"]
    for s in samples[:2]:
        n = int(rng.integers(800, 1200))
        z = rng.choice([1, 2, 3, 4], size=n, p=[0.15, 0.55, 0.25, 0.05])
        mass = rng.uniform(600, 3500, n)
        mz = (mass + z * 1.00728) / z
        rt = rng.uniform(5, 90, n)
        fwhm = rng.uniform(0.1, 0.6, n)
        apex = rng.lognormal(14, 1.2, n)
        df = pd.DataFrame({"mz": mz, "mostAbundantMz": mz + 1.00335 / z, "charge": z, "rtStart": rt - fwhm,
                           "rtApex": rt, "rtEnd": rt + fwhm, "fwhm": fwhm, "nIsotopes": rng.integers(2, 6, n),
                           "nScans": rng.integers(5, 40, n), "averagineCorr": rng.uniform(0.8, 1.0, n),
                           "mass": mass, "massCalib": mass + rng.normal(0, 0.002, n),
                           "intensityApex": apex, "intensitySum": apex * rng.uniform(3, 12, n)})[cols]
        df.to_csv(os.path.join(ddir, f"{s}.features.tsv"), sep="\t", index=False)


def _b64_floats(arr):
    return base64.b64encode(struct.pack("<%dd" % len(arr), *arr)).decode()


def build_mzml(rng, out, sample):
    """A minimal (uncompressed, 64-bit) mzML with one MS1 and one MS2 spectrum
    so the spectrum viewer and Dinosaur have something to open."""
    ms1_mz = np.sort(rng.uniform(400, 900, 60)); ms1_int = rng.lognormal(12, 1, 60)
    # b/y ions of KSTGGK[Acetyl]APR 2+ roughly (not exact masses; demo only)
    ms2_mz = np.sort(np.concatenate([rng.uniform(150, 950, 40), [129.102, 216.134, 317.182, 374.203, 431.225]]))
    ms2_int = rng.lognormal(10, 1, len(ms2_mz))

    def spectrum(idx, ms_level, mz, it, rt, precursor=None):
        prec = ""
        if precursor:
            prec = f"""<precursorList count="1"><precursor><selectedIonList count="1"><selectedIon>
<cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="{precursor[0]}" unitAccession="MS:1000040" unitName="m/z"/>
<cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="{precursor[1]}"/>
</selectedIon></selectedIonList><activation><cvParam cvRef="MS" accession="MS:1000422" name="beam-type collision-induced dissociation" value=""/></activation></precursor></precursorList>"""
        return f"""<spectrum index="{idx}" id="controllerType=0 controllerNumber=1 scan={idx+1}" defaultArrayLength="{len(mz)}">
<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{ms_level}"/>
<cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" value=""/>
<scanList count="1"><cvParam cvRef="MS" accession="MS:1000795" name="no combination" value=""/><scan>
<cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt}" unitAccession="UO:0000031" unitName="minute"/>
</scan></scanList>{prec}
<binaryDataArrayList count="2">
<binaryDataArray encodedLength="0"><cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/><cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/><cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value="" unitAccession="MS:1000040" unitName="m/z"/><binary>{_b64_floats(mz)}</binary></binaryDataArray>
<binaryDataArray encodedLength="0"><cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/><cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/><cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value="" unitAccession="MS:1000131" unitName="number of detector counts"/><binary>{_b64_floats(it)}</binary></binaryDataArray>
</binaryDataArrayList></spectrum>"""

    body = spectrum(0, 1, ms1_mz, ms1_int, 32.10) + spectrum(1, 2, ms2_mz, ms2_int, 32.12, precursor=(478.7765, 2))
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0" id="{sample}">
<cvList count="2"><cv id="MS" fullName="Proteomics Standards Initiative Mass Spectrometry Ontology" URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"/><cv id="UO" fullName="Unit Ontology" URI="http://ontologies.berkeleybop.org/uo.obo"/></cvList>
<fileDescription><fileContent><cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum" value=""/></fileContent></fileDescription>
<softwareList count="1"><software id="demo" version="1"><cvParam cvRef="MS" accession="MS:1000799" name="custom unreleased software tool" value="make_demo_data"/></software></softwareList>
<instrumentConfigurationList count="1"><instrumentConfiguration id="IC1"><cvParam cvRef="MS" accession="MS:1000031" name="instrument model" value=""/></instrumentConfiguration></instrumentConfigurationList>
<dataProcessingList count="1"><dataProcessing id="dp"><processingMethod order="0" softwareRef="demo"><cvParam cvRef="MS" accession="MS:1000544" name="Conversion to mzML" value=""/></processingMethod></dataProcessing></dataProcessingList>
<run id="{sample}" defaultInstrumentConfigurationRef="IC1">
<spectrumList count="2" defaultDataProcessingRef="dp">
{body}
</spectrumList></run></mzML>
"""
    os.makedirs(os.path.join(out, "mzML"), exist_ok=True)
    with open(os.path.join(out, "mzML", f"{sample}.mzML"), "w") as f:
        f.write(xml)


def build_ms2(rng, out, sample):
    """RawConverter-style .ms2 file with three MS2 scans (one carries the demo b-ion ladder)."""
    os.makedirs(os.path.join(out, "ms2"), exist_ok=True)
    lines = ["H\tCreationDate\tdemo", "H\tExtractor\tmake_demo_data", "H\tComments\tsynthetic"]
    ladder = [129.1022, 216.1343, 317.1819, 374.2034, 431.2249, 175.1190, 272.1717, 343.2088, 513.3143]
    for k, (scan, pmz, z) in enumerate([(2001, 478.7765, 2), (2002, 622.3410, 2), (2003, 415.2203, 3)]):
        mz = np.sort(np.concatenate([rng.uniform(150, 950, 35), ladder if k == 0 else []]))
        it = rng.lognormal(10, 1, len(mz))
        lines += [f"S\t{scan}\t{scan}\t{pmz:.4f}", f"I\tRTime\t{30 + k * 0.5:.2f}", f"Z\t{z}\t{pmz * z - (z - 1) * 1.00728:.4f}"]
        lines += [f"{a:.4f} {b:.1f}" for a, b in zip(mz, it)]
    with open(os.path.join(out, "ms2", f"{sample}.ms2"), "w") as f:
        f.write("\n".join(lines) + "\n")


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
    build_sample_folders(rng, a.out, samples)
    build_dinosaur(rng, a.out, samples)
    build_mzml(rng, a.out, samples[0])
    build_ms2(rng, a.out, samples[1])
    print(f"Demo experiment written to {a.out} ({len(samples)} samples, {len(REGIONS)} regions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
