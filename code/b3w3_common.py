"""Shared helpers for batch-3 W3 Dryad conversions (run inside Voyager Jobs)."""
import csv, hashlib, json, os, re
import numpy as np

R = os.environ.get("R", "/voyager/ceph/groups/sdp190/bpinto/ieeg-nemar-20261005")


def wtsv(p, header, rows):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(["n/a" if (v is None or v == "" or (isinstance(v, float) and np.isnan(v))) else v for v in r])


def wjson(p, o):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w") as f:
        json.dump(o, f, indent=2, ensure_ascii=False)
        f.write("\n")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def bv_name(n):
    return str(n).replace(",", "\\1")


def write_vhdr(path_noext, nch, sfreq, names, units, fmt="IEEE_FLOAT_32", resolutions=None, comment="", markers=None):
    """Write <stem>.vhdr + <stem>.vmrk for an existing <stem>.eeg (MULTIPLEXED). markers: list of (type, desc, sample0, dur_samples)."""
    stem = os.path.basename(path_noext)
    res = resolutions or [1.0] * nch
    L = ["Brain Vision Data Exchange Header File Version 1.0", f"; {comment}" if comment else "; written by b3w3 converter", "",
         "[Common Infos]", "Codepage=UTF-8", f"DataFile={stem}.eeg", f"MarkerFile={stem}.vmrk",
         "DataFormat=BINARY", "DataOrientation=MULTIPLEXED", f"NumberOfChannels={nch}",
         f"SamplingInterval={1e6 / sfreq!r}", "", "[Binary Infos]", f"BinaryFormat={fmt}", "", "[Channel Infos]"]
    for i in range(nch):
        L.append(f"Ch{i + 1}={bv_name(names[i])},,{res[i]!r},{units[i]}")
    open(path_noext + ".vhdr", "w", encoding="utf-8").write("\n".join(L) + "\n")
    M = ["Brain Vision Data Exchange Marker File, Version 1.0", "", "[Common Infos]", "Codepage=UTF-8",
         f"DataFile={stem}.eeg", "", "[Marker Infos]", "Mk1=New Segment,,1,1,0"]
    k = 2
    for typ, desc, s0, dur in (markers or []):
        M.append(f"Mk{k}={typ},{bv_name(desc)},{int(s0) + 1},{max(1, int(dur))},0")
        k += 1
    open(path_noext + ".vmrk", "w", encoding="utf-8").write("\n".join(M) + "\n")


MAT_DATE = re.compile(rb"Created on: [A-Z][a-z]{2} ([A-Z][a-z]{2}) +(\d{1,2}) (\d\d:\d\d:\d\d) (\d{4})")


def scrub_mat_header(b):
    """MAT-file 116-byte text header: 'Created on: Www Mmm dd hh:mm:ss yyyy' -> 'Created on: Mmm 01 yyyy [day->01]', same length.
    Returns (new_bytes, changed)."""
    head = b[:116]
    m = MAT_DATE.search(head)
    if not m:
        return b, False
    rep = b"Created on: " + m.group(1) + b" 01 " + m.group(4) + b" [day->01]"
    span = m.end() - m.start()
    rep = rep[:span].ljust(span, b" ")
    nh = head[:m.start()] + rep + head[m.end():]
    assert len(nh) == 116
    return nh + b[116:], True


def ym01(y, m):
    return f"{int(y):04d}-{int(m):02d}-01"


def add_electrodes_na(tree, desc, group_fn=None, types=("ECOG", "SEEG", "DBS"), spatial_ref=None):
    """Per subject(/session) ieeg dir: space-Other electrodes.tsv with x/y/z n/a (names from channels.tsv) + coordsystem.json.
    Used when the release has no coordinates, so IEEG_ELECTRODES_REQUIRED is met without inventing positions."""
    import glob
    out = []
    for sd in sorted(glob.glob(os.path.join(tree, "sub-*", "ieeg")) + glob.glob(os.path.join(tree, "sub-*", "ses-*", "ieeg"))):
        parts = sd.rstrip("/").split(os.sep)
        pref = parts[-2] if parts[-2].startswith("sub-") else f"{parts[-3]}_{parts[-2]}"
        names = []
        for ch in sorted(glob.glob(os.path.join(sd, "*_channels.tsv"))):
            with open(ch) as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    if r["type"] in types and r["name"] not in names:
                        names.append(r["name"])
        wtsv(os.path.join(sd, f"{pref}_space-Other_electrodes.tsv"), ["name", "x", "y", "z", "size", "group"],
             [[n, "n/a", "n/a", "n/a", "n/a", group_fn(n) if group_fn else "n/a"] for n in names])
        if spatial_ref:
            wjson(os.path.join(sd, f"{pref}_space-Other_electrodes.json"), {"SpatialReference": spatial_ref})
        wjson(os.path.join(sd, f"{pref}_space-Other_coordsystem.json"), {"iEEGCoordinateSystem": "Other", "iEEGCoordinateUnits": "n/a",
                                                                         "iEEGCoordinateSystemDescription": desc})
        out.append((pref, len(names)))
    return out
