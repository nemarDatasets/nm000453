"""Shared helpers for batch-3 W10 (wave 3) conversions, run inside Voyager Jobs. Builds on b3w3_common.
write_run(): one continuous recording -> BrainVision (IEEE float32, MULTIPLEXED) + channels.tsv + ieeg.json + vmrk markers,
with a read-back check through MNE. mat_strings(): every string stored in a .mat (scipy / h5py) for the privacy review."""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b3w3_common import wtsv, wjson, write_vhdr, sha256_file, scrub_mat_header, ym01, add_electrodes_na  # noqa: F401

CH_COLS = ["name", "type", "units", "low_cutoff", "high_cutoff", "sampling_frequency", "reference", "group", "status", "status_description", "description"]


def write_run(D, stem, X, sfreq, names, units, types, sidecar, ch_extra=None, markers=None, comment="", scale=None, check=True):
    """X: (n_samples, n_ch) array in the physical unit `units` (or raw ints with `scale` = per-channel factors to the unit).
    Writes D/stem_ieeg.{eeg,vhdr,vmrk,json} + D/stem_channels.tsv. ch_extra: list of dicts (per channel) overriding CH_COLS.
    Returns dict(n, max_abs_err, roundtrip)."""
    os.makedirs(D, exist_ok=True)
    X = np.asarray(X)
    n, nch = X.shape
    if scale is not None:
        Xp = X.astype(np.float64) * np.asarray(scale, dtype=np.float64)[None, :]
    else:
        Xp = X.astype(np.float64)
    X32 = Xp.astype(np.float32)
    fin = np.isfinite(Xp)
    err = float(np.max(np.abs(X32[fin].astype(np.float64) - Xp[fin]))) if fin.any() else 0.0
    base = os.path.join(D, stem + "_ieeg")
    np.ascontiguousarray(X32).astype("<f4").tofile(base + ".eeg")
    write_vhdr(base, nch, sfreq, names, units, comment=comment, markers=markers)
    rows = []
    for i, c in enumerate(names):
        r = {"name": c, "type": types[i], "units": units[i], "low_cutoff": "n/a", "high_cutoff": "n/a", "sampling_frequency": sfreq,
             "reference": "n/a", "group": "n/a", "status": "good", "status_description": "n/a", "description": "n/a"}
        if ch_extra:
            r.update(ch_extra[i])
        rows.append([r[k] for k in CH_COLS])
    wtsv(os.path.join(D, stem + "_channels.tsv"), CH_COLS, rows)
    sc = dict(sidecar)
    sc.setdefault("SamplingFrequency", sfreq)
    sc.setdefault("RecordingDuration", round(n / sfreq, 6))
    sc.setdefault("RecordingType", "continuous")
    for key, t in (("ECOGChannelCount", "ECOG"), ("SEEGChannelCount", "SEEG"), ("DBSChannelCount", "DBS"), ("EEGChannelCount", "EEG"),
                   ("EMGChannelCount", "EMG"), ("MiscChannelCount", "MISC"), ("TriggerChannelCount", "TRIG")):
        sc.setdefault(key, sum(1 for x in types if x == t))
    wjson(base + ".json", sc)
    rt = None
    if check:
        import mne
        rr = mne.io.read_raw_brainvision(base + ".vhdr", preload=False, verbose="error")
        k = min(5000, n)
        A = rr.get_data(start=0, stop=k)
        # MNE scales by unit (µV -> V etc.); compare up to a per-channel factor of 10^k
        B = X32[:k].T.astype(np.float64)
        ok = rr.n_times == n and len(rr.ch_names) == nch
        if ok:
            for i in range(nch):
                b = B[i]; a = A[i]
                m = np.isfinite(b) & (np.abs(b) > 0)
                if not m.any():
                    continue
                f = np.median(a[m] / b[m])
                ok = ok and np.allclose(a[np.isfinite(b)], f * b[np.isfinite(b)], rtol=1e-5, atol=1e-12) and abs(np.log10(abs(f)) - round(np.log10(abs(f)))) < 1e-6
        rt = bool(ok)
    return {"n_samples": int(n), "sfreq": sfreq, "n_channels": nch, "float32_max_abs_err": err, "roundtrip_ok": rt, "duration_s": n / sfreq}


def mat_strings(path, limit=5000):
    """Return the set of strings stored in a .mat file (variable names excluded), for the privacy review."""
    out = set()
    def walk(v, depth=0):
        if len(out) > limit or depth > 12:
            return
        if isinstance(v, str):
            out.add(v); return
        if isinstance(v, bytes):
            out.add(v.decode("latin-1")); return
        if isinstance(v, np.ndarray):
            if v.dtype.kind in "US":
                for x in v.flat:
                    out.add(str(x))
            elif v.dtype.names:
                for x in v.flat:
                    for f in v.dtype.names:
                        walk(x[f], depth + 1)
            elif v.dtype == object:
                for x in v.flat:
                    walk(x, depth + 1)
    try:
        import scipy.io as sio
        m = sio.loadmat(path, squeeze_me=False, struct_as_record=True, verify_compressed_data_integrity=False)
        for k, v in m.items():
            if not k.startswith("__"):
                walk(v)
        out.add("__header__:" + m.get("__header__", b"").decode("latin-1"))
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as f:
            def vis(n, o):
                if isinstance(o, h5py.Dataset) and o.attrs.get("MATLAB_class", b"") == b"char" and o.size < 100000:
                    try:
                        out.add("".join(chr(c) for c in o[()].flatten(order="F")))
                    except Exception:
                        pass
            f.visititems(vis)
    return out


ID_PAT = re.compile(r"(\b(19|20)\d\d[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])\b)|(\b(0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])[-/.](19|20)?\d\d\b)|"
                    r"(\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun) [A-Z][a-z]{2} +\d{1,2} )|(@)|(\b(MRN|DOB|patient name|birth)\b)", re.I)


def flag_strings(strings):
    return sorted(s[:200] for s in strings if ID_PAT.search(s))
