"""External tool integration: Dinosaur MS1 feature detection (Java).

Install:  python ep_tools.py install              (downloads tools/Dinosaur-<ver>.free.jar)
Run:      python ep_tools.py run <file.mzML> [...]    (writes <file>.features.tsv next to the input)
Annotator: python ep_tools.py annotator [file.mgf]    (locate / launch the Annotator desktop app)

From the dashboard the same functions are used to launch Dinosaur in the
background on every mzML found in an experiment directory.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(HERE, "tools")
DINOSAUR_VERSION = os.environ.get("DINOSAUR_VERSION", "1.2.0")
DINOSAUR_URL = f"https://github.com/fickludd/dinosaur/releases/download/{DINOSAUR_VERSION}/Dinosaur-{DINOSAUR_VERSION}.free.jar"
DINOSAUR_JAR = os.path.join(TOOLS_DIR, f"Dinosaur-{DINOSAUR_VERSION}.free.jar")


def java_available() -> str | None:
    return shutil.which("java")


def dinosaur_jar() -> str | None:
    """Path to an installed Dinosaur jar (env DINOSAUR_JAR, tools/, or None)."""
    env = os.environ.get("DINOSAUR_JAR")
    if env and os.path.isfile(env):
        return env
    if os.path.isfile(DINOSAUR_JAR):
        return DINOSAUR_JAR
    if os.path.isdir(TOOLS_DIR):
        for fn in sorted(os.listdir(TOOLS_DIR)):
            if fn.lower().startswith("dinosaur") and fn.endswith(".jar"):
                return os.path.join(TOOLS_DIR, fn)
    return None


def install_dinosaur(force: bool = False) -> str:
    os.makedirs(TOOLS_DIR, exist_ok=True)
    if os.path.isfile(DINOSAUR_JAR) and not force:
        return DINOSAUR_JAR
    tmp = DINOSAUR_JAR + ".part"
    urllib.request.urlretrieve(DINOSAUR_URL, tmp)
    os.replace(tmp, DINOSAUR_JAR)
    return DINOSAUR_JAR


def dinosaur_command(mzml: str, out_dir: str | None = None, concurrency: int = 2, extra=None) -> list[str]:
    jar = dinosaur_jar()
    if not jar:
        raise FileNotFoundError("Dinosaur jar not installed; run `python ep_tools.py install`")
    cmd = [java_available() or "java", "-jar", jar, f"--concurrency={int(concurrency)}", "--verbose", "--nReport=0"]
    if out_dir:
        cmd.append(f"--outDir={out_dir}")
    if extra:
        cmd.extend(extra)
    cmd.append(mzml)
    return cmd


def run_dinosaur(mzml: str, out_dir: str | None = None, concurrency: int = 2, timeout: int | None = None) -> dict:
    """Run Dinosaur synchronously. Returns dict(ok, returncode, stdout, stderr, features, seconds)."""
    out_dir = out_dir or os.path.dirname(os.path.abspath(mzml))
    os.makedirs(out_dir, exist_ok=True)
    cmd = dinosaur_command(mzml, out_dir, concurrency)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    base = os.path.splitext(os.path.basename(mzml))[0]
    feats = os.path.join(out_dir, base + ".features.tsv")
    return {"ok": proc.returncode == 0 and os.path.isfile(feats), "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:],
            "features": feats if os.path.isfile(feats) else None, "seconds": round(time.time() - t0, 1),
            "command": " ".join(cmd)}


class DinosaurJob:
    """Background runner over several mzML files with a progress log the
    dashboard can poll."""

    def __init__(self, files, out_dir=None, concurrency=2):
        self.files = list(files)
        self.out_dir = out_dir
        self.concurrency = concurrency
        self.log: list[str] = []
        self.results: list[dict] = []
        self.done = False
        self.started = None
        self._thread = None

    def start(self):
        self.started = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        for f in self.files:
            self.log.append(f"[{time.strftime('%H:%M:%S')}] Dinosaur: {os.path.basename(f)}")
            try:
                r = run_dinosaur(f, self.out_dir, self.concurrency)
            except Exception as e:  # pragma: no cover
                r = {"ok": False, "stderr": str(e), "features": None, "seconds": 0}
            self.results.append(r)
            status = "OK" if r["ok"] else "FAILED"
            self.log.append(f"    -> {status} in {r.get('seconds', 0)}s" + (f": {r['features']}" if r.get("features") else ""))
            if not r["ok"] and r.get("stderr"):
                self.log.append("    " + r["stderr"].strip().splitlines()[-1][:300] if r["stderr"].strip() else "")
        self.done = True
        self.log.append(f"[{time.strftime('%H:%M:%S')}] finished {sum(1 for r in self.results if r['ok'])}/{len(self.files)} files")

    @property
    def progress(self):
        return len(self.results), len(self.files)


# ----------------------------------------------------------------------
# Annotator (Snijder lab, https://github.com/snijderlab/annotator): desktop
# spectrum annotator built on rustyms. Opens mzML / MGF / Thermo RAW / Bruker.
# The dashboard exports MGF files it can open and uses the rustyms Python
# bindings (pip install rustyms==0.9.0) as an in-process engine.
# ----------------------------------------------------------------------
ANNOTATOR_URL = "https://github.com/snijderlab/annotator/releases/latest"


def annotator_binary() -> str | None:
    env = os.environ.get("ANNOTATOR_BIN")
    if env and os.path.isfile(env):
        return env
    for name in ("annotator", "Annotator", "annotator.exe", "Annotator.exe", "annotator.AppImage", "Annotator.AppImage"):
        p = shutil.which(name)
        if p:
            return p
        p = os.path.join(TOOLS_DIR, name)
        if os.path.isfile(p):
            return p
    if os.path.isdir(TOOLS_DIR):
        for fn in sorted(os.listdir(TOOLS_DIR)):
            if fn.lower().startswith("annotator") and not fn.endswith((".jar", ".part")):
                return os.path.join(TOOLS_DIR, fn)
    return None


def launch_annotator(path: str | None = None) -> dict:
    """Start the Annotator desktop app (optionally with a file). Non-blocking."""
    exe = annotator_binary()
    if not exe:
        return {"ok": False, "message": f"Annotator binary not found. Download it from {ANNOTATOR_URL} into tools/ "
                                        "or set ANNOTATOR_BIN. On Linux give the AppImage execute permission."}
    cmd = [exe] + ([path] if path else [])
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        return {"ok": True, "message": "Annotator launched: " + " ".join(cmd)}
    except Exception as e:
        return {"ok": False, "message": f"Could not launch Annotator: {e}"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    if argv[0] == "install":
        print("Installed", install_dinosaur(force="--force" in argv)); return 0
    if argv[0] == "annotator":
        exe = annotator_binary()
        print("Annotator:", exe or f"not installed. Download from {ANNOTATOR_URL} into {TOOLS_DIR}")
        if exe and len(argv) > 1:
            print(launch_annotator(argv[1])["message"])
        return 0
    if argv[0] == "run":
        files = [a for a in argv[1:] if not a.startswith("--")]
        if not files:
            print("usage: ep_tools.py run <file.mzML> [...]"); return 2
        if not dinosaur_jar():
            install_dinosaur()
        rc = 0
        for f in files:
            r = run_dinosaur(f)
            print(("OK  " if r["ok"] else "FAIL"), f, "->", r["features"] or r["stderr"][-200:])
            rc |= 0 if r["ok"] else 1
        return rc
    print("unknown command", argv[0]); return 2


if __name__ == "__main__":
    sys.exit(main())
