"""d3-gdansk-mer (MOST Wiedzy doi:10.34808/nq8v-t162, CC0-1.0; Osowska, Szymanski, Libionka, Sci Data 13:870, 2026)
-> BIDS-iEEG DERIVATIVE dataset (F2, Bruno 2026-10-08: short artefact-cleaned segments count).

Release (dataset.zip): MER/<P###>/depth<d>_kanal<label>.csv (Time, '2: preprocessed'[, '1: raw']) + protokoll.txt,
brain_layer_annotations/<structure>/<P###>/depth<d>_kanal<label>.csv (Time, '1: raw', '2: preprocessed'), metadata.csv.
Mapping: one BIDS run per CSV.
  MER/<P>/<f>                        -> sub-<P>/ieeg/sub-<P>_task-mer_acq-<label>_run-<k>_ieeg.vhdr
  brain_layer_annotations/<S>/<P>/<f> -> sub-<P>/ieeg/sub-<P>_task-merannot_acq-<label>_run-<k>_ieeg.vhdr
  acq = electrode label from the file name, lower-case alphanumerics (e.g. Central -> central, 3Central -> 3central).
  run k = order of the recording site in protokoll.txt (first 'New site no ... Depth' line with that depth), then depth.
Channels: '<label>' = '2: preprocessed' (authors' 4th-order Butterworth band-pass 500-5000 Hz, clipped to +-500 instrument
units); '<label>_raw' = '1: raw' where present (int16 instrument values decoded from the .dat files, unfiltered).
Data: BrainVision IEEE float32. The CSV text is parsed as float64 (round_trip); float32 cast deviation is measured and
every .eeg is read back and compared with float32(CSV) bit for bit; the raw column is integer valued (exact in float32).
Usage: python f2_gdansk_convert.py <work_dir> <bids_root> [nproc]
Env ONLY=P001,P106 restricts the subjects (smoke).
"""
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b3w3_common import wjson, wtsv  # noqa: E402
from b3w10_common import write_run  # noqa: E402

W, OUT = sys.argv[1], sys.argv[2]
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else 16
REL = os.path.join(W, "extracted", "DBS_dataset", "DBS_dataset_v3")
DS = os.path.join(REL, "dataset", "dataset")
CODES = os.path.join(REL, "codes.zip")
FS = 20000.0
DOI = "10.34808/nq8v-t162"
ONLY = [x for x in os.environ.get("ONLY", "").split(",") if x]
FN = re.compile(r"^depth(-?\d+,\d+)_kanal([A-Za-z0-9]+)\.csv$")
STRUCT = {"Internal Globus Pallidus": "GPi", "External Globus Pallidus": "GPe", "Striatum or Putamen": "STR_or_putamen"}
# the 4th entry of the authors' parser folder-exclusion list (data_transformer.py) is a first name: redacted in sourcedata
REDACT_RE = re.compile(rb'(\["awegaas", "empty", "test", ")([^"]+)("\])')


def depth_f(s):
    return float(s.replace(",", "."))


def scrub_protocol(txt):
    n = 0
    def iso(m):
        nonlocal n
        n += 1
        return f"{m.group(1)}-{m.group(2)}-01"
    txt = re.sub(r"\b((?:19|20)\d\d)-(0[1-9]|1[0-2])-(\d\d)\b", iso, txt)
    def dmy(m):
        nonlocal n
        n += 1
        return f"01{m.group(2)}{m.group(3)}{m.group(2)}{m.group(4)}"
    txt = re.sub(r"\b(\d\d)([./])(0[1-9]|1[0-2])\2((?:19|20)\d\d)\b", dmy, txt)
    # numeric archive folder / file ids in .dat paths (may encode acquisition times) -> <id>
    txt, k1 = re.subn(r"(patienten\\[^\\\s]+\\)\d{5,}(\\)", r"\1<id>\2", txt)
    txt, k2 = re.subn(r"(_(?:MER_CH\d+|TRG|STIM|EMG\w*))_\d{5,}(\.dat)", r"\1_<id>\2", txt)
    return txt, {"dates": n, "dat_ids": k1 + k2}


def sites(txt):
    """protokoll -> list of (global_index, site_no, depth_float) in file order."""
    out = []
    for i, m in enumerate(re.finditer(r"New site no (\d+) created\( Depth: (-?\d+[.,]\d+) mm \)", txt)):
        out.append((i, int(m.group(1)), depth_f(m.group(2))))
    return out


def read_csv(p):
    df = pd.read_csv(p, float_precision="round_trip")
    cols = list(df.columns)
    t = df["Time"].to_numpy(np.float64)
    pre = df["2: preprocessed"].to_numpy(np.float64)
    raw = df["1: raw"].to_numpy(np.float64) if "1: raw" in cols else None
    return cols, t, pre, raw


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def do_subject(P):
    sub = f"sub-{P}"
    D = os.path.join(OUT, sub, "ieeg")
    os.makedirs(D, exist_ok=True)
    rep = {"sub": sub, "recordings": [], "notes": []}
    pp = os.path.join(DS, "MER", P, "protokoll.txt")
    ptxt = open(pp, encoding="utf-8", errors="replace").read() if os.path.exists(pp) else ""
    if not ptxt:
        rep["notes"].append(f"{P}: no protokoll.txt")
    st = sites(ptxt)
    first = {}
    for gi, sn, d in st:
        first.setdefault(round(d, 3), (gi, sn))
    nsite = Counter(round(d, 3) for _, _, d in st)
    # inventory: main + annotation files
    items = []
    for f in sorted(os.listdir(os.path.join(DS, "MER", P))):
        if f == "protokoll.txt":
            continue
        m = FN.match(f)
        if not m:
            rep["notes"].append(f"{P}: unexpected file {f}")
            continue
        items.append(("mer", None, f, os.path.join(DS, "MER", P, f), m.group(1), m.group(2)))
    for sname in sorted(STRUCT):
        d = os.path.join(DS, "brain_layer_annotations", sname, P)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                m = FN.match(f)
                if not m:
                    rep["notes"].append(f"{P}: unexpected annotation file {sname}/{f}")
                    continue
                items.append(("merannot", sname, f, os.path.join(d, f), m.group(1), m.group(2)))
    # run numbering per (task, acq): protocol order, then depth
    groups = defaultdict(list)
    for it in items:
        acq = it[5].lower()
        groups[(it[0], acq)].append(it)
    plan = []
    for (task, acq), lst in groups.items():
        lst.sort(key=lambda it: (first.get(round(depth_f(it[4]), 3), (10 ** 9, None))[0], depth_f(it[4]), it[1] or ""))
        for k, it in enumerate(lst, start=1):
            plan.append((task, acq, k, it))
    # load annotations first so the main runs can carry their intervals as events
    annot_ivals = defaultdict(list)  # (label, depthstr) -> [(t0, t1, struct, annot_stem, pre_vector)]
    scans = []
    results = {}
    for task, acq, k, it in sorted(plan, key=lambda x: (x[0] != "merannot", x[1], x[2])):
        _, sname, f, path, dstr, label = it
        cols, t, pre, raw = read_csv(path)
        n = len(t)
        dt = np.diff(t)
        fs_est = 1.0 / float(np.median(dt)) if n > 1 else FS
        gap = float(np.max(np.abs(dt - 1.0 / FS))) if n > 1 else 0.0
        contiguous = bool(gap < 1e-6)
        if abs(fs_est - FS) > 1e-3 * FS:
            rep["notes"].append(f"{P}/{f}: sampling rate estimate {fs_est}")
        stem = f"{sub}_task-{task}_acq-{acq}_run-{k:03d}"
        names = [label] + ([f"{label}_raw"] if raw is not None else [])
        X = np.stack([pre] + ([raw] if raw is not None else []), axis=1)
        traj = re.sub(r"\d", "", label).lower()
        ambiguous = label == "Central3Lateral"
        elab = (f"electrode label '{label}' as released (protocol line '2: Central3: Lateral' is missing a comma; Central or Lateral cannot be told apart)" if ambiguous
                else f"Ben-gun microelectrode '{label}' ({traj}" + (f"; the digit is the amplifier channel number kept by the authors' parser, which strips only '1' and '2'" if re.search(r"\d", label) else "") + ")")
        ch_extra = [{"type": "DBS", "units": "µV", "low_cutoff": 500, "high_cutoff": 5000,
                     "reference": "scalp reference electrode (article)", "group": label,
                     "description": f"{elab}; authors' '2: preprocessed' column: 4th-order Butterworth band-pass 500-5000 Hz (scipy lfilter), clipped to +-500; instrument units interpreted by the authors as µV-equivalent (unknown global scale)"}]
        if raw is not None:
            ch_extra.append({"type": "DBS", "units": "µV", "low_cutoff": "n/a", "high_cutoff": "n/a",
                             "reference": "scalp reference electrode (article)", "group": label,
                             "description": f"{elab}; authors' '1: raw' column: int16 values decoded from the acquisition .dat file, no software filter, no overflow correction; instrument units (µV-equivalent per the authors, unknown global scale)"})
        dur = n / FS
        sidecar = {
            "TaskName": task,
            "TaskDescription": ("Intraoperative microelectrode recording (MER) at one depth of a Ben-gun microelectrode trajectory during awake DBS surgery; "
                                "this run is the single segment the authors' signal-quality classifier kept from that recording (not the full recording)") if task == "mer" else
                               ("Excerpt of a MER recording labelled by an experienced electrophysiologist as representative of a basal ganglia structure "
                                "(striatum/putamen, GPe or GPi) along a GPi-targeted trajectory (authors' brain_layer_annotations)"),
            "Instructions": "n/a (awake surgery under local anaesthesia; no task)",
            "SamplingFrequency": FS, "PowerLineFrequency": 50,
            "SoftwareFilters": {"ButterworthBandpass": {"Description": "Authors' parser (codes/parser/utils.py): overflow correction, 4th-order Butterworth band-pass 500-5000 Hz (scipy.signal.lfilter, causal), amplitude clipped to +-500 instrument units; applies to the preprocessed channel only", "LowCutoff": 500, "HighCutoff": 5000, "Order": 4}},
            "HardwareFilters": {"AcquisitionSystem": {"Description": "ISIS MER hardware band-pass; per-site settings (e.g. HW-fcHP 0 Hz, HW-fcLP 5000 Hz) are in the protocol files in sourcedata"}},
            "Manufacturer": "inomed Medizintechnik GmbH", "ManufacturersModelName": "ISIS MER system (MicroMacro electrodes, Ben-gun MicroDrive)",
            "iEEGReference": "Scalp reference electrode (article)", "iEEGGround": "Chest (article)",
            "iEEGPlacementScheme": "Up to five parallel microelectrodes in a Ben-gun configuration (central, anterior, posterior, medial, lateral) advanced along the planned trajectory; depth relative to the reference target (0 mm = target)",
            "iEEGElectrodeGroups": "One group per microelectrode label",
            "ElectrodeManufacturer": "inomed Medizintechnik GmbH",
            "DBSChannelCount": len(names), "SEEGChannelCount": 0, "ECOGChannelCount": 0,
            "RecordingType": "epoched", "EpochLength": round(dur, 6), "RecordingDuration": round(dur, 6),
            "ElectricalStimulation": False,
            "SourceFile": os.path.relpath(path, os.path.join(REL, "dataset")),
            "SourceTimeStart": round(float(t[0]), 6),
            "DepthFromTarget_mm": depth_f(dstr),
        }
        res = write_run(D, stem, X, FS, names, ["µV"] * len(names), ["DBS"] * len(names), sidecar, ch_extra=ch_extra,
                        comment=f"f2_gdansk_convert.py: {sidecar['SourceFile']} as float32")
        back = np.fromfile(os.path.join(D, stem + "_ieeg.eeg"), dtype="<f4").reshape(-1, len(names))
        exact32 = bool(back.shape == X.shape and np.array_equal(back, X.astype(np.float32)))
        err64 = float(np.max(np.abs(back.astype(np.float64) - X)))
        raw_int = bool(raw is None or np.array_equal(raw, np.round(raw)))
        raw_exact = bool(raw is None or np.array_equal(back[:, 1].astype(np.float64), raw))
        clip_ok = bool(np.max(np.abs(pre)) <= 500.0 + 1e-6)
        site = first.get(round(depth_f(dstr), 3))
        ev = [[0.0, round(dur, 6), "mer_segment" if task == "mer" else f"annotated_{STRUCT[sname]}", 0, depth_f(dstr), round(float(t[0]), 6), "n/a"]]
        if task == "merannot":
            annot_ivals[(label, dstr)].append((float(t[0]), float(t[-1]), STRUCT[sname], stem, pre))
        else:
            for (a0, a1, sl, ast, apre) in annot_ivals.get((label, dstr), []):
                on = int(round((a0 - t[0]) * FS))
                same = None
                if 0 <= on and on + len(apre) <= n:
                    same = bool(np.array_equal(pre[on:on + len(apre)], apre))
                    ev.append([round(on / FS, 6), round(len(apre) / FS, 6), f"annotated_{sl}", on, depth_f(dstr), round(a0, 6), f"{ast}_ieeg.vhdr"])
                rep["notes"].append(f"{P}/{f}: annotation {ast} at {a0:.4f}-{a1:.4f} s; inside this segment: {0 <= on and on + len(apre) <= n}; identical preprocessed values: {same}")
        wtsv(os.path.join(D, stem + "_events.tsv"), ["onset", "duration", "trial_type", "sample", "depth_mm", "source_time_s", "annotation_run"], ev)
        scans.append([f"ieeg/{stem}_ieeg.vhdr", depth_f(dstr), label, traj if not ambiguous else "central_or_lateral",
                      site[1] if site else "n/a", nsite.get(round(depth_f(dstr), 3), 0), sidecar["SourceFile"],
                      round(float(t[0]), 6), round(float(t[-1]), 6), n, "yes" if raw is not None else "no",
                      STRUCT[sname] if task == "merannot" else "n/a"])
        rr = dict(sub=sub, stem=stem, src=sidecar["SourceFile"], cols=cols, n=int(n), dur=dur, contiguous=contiguous, max_dt_dev=gap,
                  exact_float32=exact32, max_abs_err_vs_csv_float64=err64, raw_integer=raw_int, raw_exact=raw_exact, clip_ok=clip_ok,
                  mne_ok=res["roundtrip_ok"], has_raw=raw is not None, site_no=site[1] if site else None,
                  sha256=sha256(path), bytes=os.path.getsize(path))
        rr["roundtrip_ok"] = bool(exact32 and raw_exact and res["roundtrip_ok"] and err64 <= 1e-4 * max(1.0, float(np.max(np.abs(X)))))
        rep["recordings"].append(rr)
    labs = sorted({it[5] for it in items})
    erows = []
    for lab in labs:
        dd = sorted({depth_f(it[4]) for it in items if it[5] == lab})
        erows.append([lab, "n/a", "n/a", "n/a", "n/a", lab, "microelectrode",
                      "central_or_lateral" if lab == "Central3Lateral" else re.sub(r"\d", "", lab).lower(), f"{dd[0]:g}", f"{dd[-1]:g}"])
    wtsv(os.path.join(D, f"{sub}_space-Other_electrodes.tsv"), ["name", "x", "y", "z", "size", "group", "type", "trajectory", "min_depth_mm", "max_depth_mm"], erows)
    wjson(os.path.join(D, f"{sub}_space-Other_electrodes.json"), {
        "name": {"Description": "Microelectrode label as in the released file names (channel name of the runs; '<label>_raw' channels come from the same electrode)"},
        "x": {"Description": "Not released (n/a)", "Units": "mm"}, "y": {"Description": "Not released (n/a)", "Units": "mm"}, "z": {"Description": "Not released (n/a)", "Units": "mm"},
        "type": {"Description": "microelectrode (inomed MicroMacro, Ben-gun array)"},
        "trajectory": {"Description": "Ben-gun position: central, anterior, posterior, medial or lateral"},
        "min_depth_mm": {"Description": "Shallowest released recording depth relative to the reference target", "Units": "mm"},
        "max_depth_mm": {"Description": "Deepest released recording depth relative to the reference target", "Units": "mm"},
        "SpatialReference": "none: no coordinates were released; position is given only as depth relative to the planned target along each Ben-gun trajectory (scans.tsv depth_mm)"})
    wjson(os.path.join(D, f"{sub}_space-Other_coordsystem.json"), {
        "iEEGCoordinateSystem": "Other", "iEEGCoordinateUnits": "mm",
        "iEEGCoordinateSystemDescription": "No electrode coordinates were released. Each microelectrode position is only known as its Ben-gun position (central/anterior/posterior/medial/lateral, 2 mm spacing in a standard Ben-gun) and the recording depth in mm relative to the MRI-planned target along the trajectory (0 = target, negative = above). Coordinates are n/a."})
    wtsv(os.path.join(OUT, sub, f"{sub}_scans.tsv"),
         ["filename", "depth_mm", "electrode_label", "trajectory", "protocol_site_no", "protocol_sites_at_depth", "source_file",
          "source_time_start_s", "source_time_end_s", "n_samples", "has_raw_channel", "annotated_structure"], sorted(scans))
    return rep, ptxt


def main():
    meta = list(csv.reader(open(os.path.join(DS, "metadata.csv"), encoding="utf-8-sig"), delimiter=";"))
    hdr, rows = meta[0], [r for r in meta[1:] if r and r[0].strip()]
    assert hdr == ["ID", "DATE", "DIAGNOSIS", "GENDER", "AGE", "TARGET"], hdr
    pats = sorted(os.listdir(os.path.join(DS, "MER")))
    ann_p = sorted({p for s in STRUCT for p in os.listdir(os.path.join(DS, "brain_layer_annotations", s))})
    assert set(ann_p) <= set(pats), ann_p
    if ONLY:
        pats = [p for p in pats if p in ONLY]
    os.makedirs(OUT, exist_ok=True)
    with Pool(NPROC) as pool:
        outs = pool.map(do_subject, pats, chunksize=1)
    reps = [o[0] for o in outs]
    recs = [r for rp in reps for r in rp["recordings"]]
    notes = [n for rp in reps for n in rp["notes"]]
    # participants
    md = {r[0].strip(): r for r in rows}
    sexmap = {"M": "M", "F": "F"}
    parts = []
    for p in pats:
        r = md.get(p)
        if r is None:
            notes.append(f"{p}: not in metadata.csv")
            parts.append([f"sub-{p}", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        _, date, dx, sx, age, tgt = [x.strip() for x in r]
        assert re.fullmatch(r"(\d{4}-\d\d)?", date), date
        parts.append([f"sub-{p}", age or "n/a", sexmap.get(sx, "n/a") if sx else "n/a", dx or "n/a", tgt or "n/a", date or "n/a"])
    wtsv(os.path.join(OUT, "participants.tsv"), ["participant_id", "age", "sex", "diagnosis_icd10", "dbs_target", "procedure_year_month"], parts)
    wjson(os.path.join(OUT, "participants.json"), {
        "age": {"Description": "Age at the time of the procedure (metadata.csv AGE)", "Units": "year"},
        "sex": {"Description": "metadata.csv GENDER", "Levels": {"M": "male", "F": "female"}},
        "diagnosis_icd10": {"Description": "ICD-10 code (metadata.csv DIAGNOSIS): G20 Parkinson's disease, G10 Huntington's disease, G24 dystonia, G25 other extrapyramidal and movement disorders, G40 epilepsy, G22 parkinsonism in diseases classified elsewhere, G29 other degenerative diseases of the nervous system, R25 abnormal involuntary movements; n/a = not retrievable from the archive (article)"},
        "dbs_target": {"Description": "metadata.csv TARGET: surgical target. 'A_AND_B' = recording spanned two structures (upper/lower part of the microelectrode); STN_OR_GPi (G20/G22) = STN or GPi, about 90% STN per the article"},
        "procedure_year_month": {"Description": "Procedure date, year and month only (metadata.csv DATE)"}})
    wjson(os.path.join(OUT, "task-mer_events.json"), {
        "trial_type": {"Description": "Event type", "Levels": {
            "mer_segment": "the whole run: the segment of the MER recording kept by the authors' signal-quality classifier",
            "annotated_GPi": "interval labelled internal globus pallidus by the authors' electrophysiologist (brain_layer_annotations)",
            "annotated_GPe": "interval labelled external globus pallidus", "annotated_STR_or_putamen": "interval labelled striatum or putamen"}},
        "sample": {"Description": "Onset sample (0-based, 20 kHz)"},
        "depth_mm": {"Description": "Recording depth relative to the reference target (0 = target, negative = above target)", "Units": "mm"},
        "source_time_s": {"Description": "Value of the CSV 'Time' column at the event onset: seconds from the start of the original (unreleased) site recording", "Units": "s"},
        "annotation_run": {"Description": "task-merannot run holding the annotated excerpt"}})
    shutil.copy(os.path.join(OUT, "task-mer_events.json"), os.path.join(OUT, "task-merannot_events.json"))
    scans_json = {
        "depth_mm": {"Description": "Depth relative to the reference target from the file name (0 = target, negative = above, positive = beyond)", "Units": "mm"},
        "electrode_label": {"Description": "Microelectrode label from the file name ('kanal<label>')"},
        "trajectory": {"Description": "Ben-gun position (label without the amplifier channel digit); central_or_lateral for the ambiguous 'Central3Lateral' label"},
        "protocol_site_no": {"Description": "'New site no' of the first protocol entry with this depth (protokoll.txt); n/a if none"},
        "protocol_sites_at_depth": {"Description": "Number of protocol site entries with this depth (more than 1 when the protocol holds several sessions)"},
        "source_file": {"Description": "CSV in the MOST Wiedzy dataset.zip (path below 'dataset/')"},
        "source_time_start_s": {"Description": "First CSV 'Time' value: offset of the kept segment within the original site recording", "Units": "s"},
        "source_time_end_s": {"Description": "Last CSV 'Time' value", "Units": "s"},
        "n_samples": {"Description": "Samples in the run (20 kHz)"},
        "has_raw_channel": {"Description": "yes if the CSV has the '1: raw' column (stored as channel <label>_raw)"},
        "annotated_structure": {"Description": "task-merannot runs: structure folder of brain_layer_annotations (GPi, GPe, STR_or_putamen)"}}
    for p in parts:
        wjson(os.path.join(OUT, p[0], f"{p[0]}_scans.json"), scans_json)
    # ---- sourcedata ----
    S = os.path.join(OUT, "sourcedata", "most_wiedzy_nq8v-t162")
    os.makedirs(os.path.join(S, "protocols"), exist_ok=True)
    prot = {}
    for P, (rp, ptxt) in zip(pats, outs):
        if ptxt:
            s, c = scrub_protocol(ptxt)
            open(os.path.join(S, "protocols", f"{P}_protokoll.txt"), "w", encoding="utf-8").write(s)
            prot[P] = c
    shutil.copy(os.path.join(DS, "metadata.csv"), os.path.join(S, "metadata.csv"))
    shutil.copy(os.path.join(DS, "readme.md"), os.path.join(S, "readme.md"))
    with open(os.path.join(S, "file_map.tsv"), "w") as fh:
        fh.write("source_file\tbids_file\tsource_columns\tsource_bytes\tsource_sha256\tn_samples\tduration_s\n")
        for r in sorted(recs, key=lambda r: r["src"]):
            fh.write(f"{r['src']}\t{r['sub']}/ieeg/{r['stem']}_ieeg.vhdr\t{'|'.join(r['cols'])}\t{r['bytes']}\t{r['sha256']}\t{r['n']}\t{r['dur']:.5f}\n")
    zred = []
    with zipfile.ZipFile(CODES) as zi, zipfile.ZipFile(os.path.join(S, "codes.zip"), "w", compression=zipfile.ZIP_DEFLATED) as zo:
        for it in zi.infolist():
            if it.filename.endswith(".pyc"):
                zred.append(f"dropped {it.filename} (compiled cache)")
                continue
            b = zi.read(it.filename)
            if it.filename.endswith("data_transformer.py"):
                b, k = REDACT_RE.subn(rb"\1[REDACTED]\3", b)
                zred.append(f"redacted {k} first name(s) in the folder-exclusion list of {it.filename}")
            y, mo = it.date_time[0], it.date_time[1]
            ni = zipfile.ZipInfo(it.filename, date_time=(max(y, 1980), mo, 1, 0, 0, 0))
            ni.compress_type = zipfile.ZIP_STORED if it.is_dir() else zipfile.ZIP_DEFLATED
            ni.external_attr = it.external_attr
            zo.writestr(ni, b)
    open(os.path.join(S, "README_sourcedata.txt"), "w").write(
        "Files from the MOST Wiedzy release doi:10.34808/nq8v-t162 (CC0-1.0), archive DBS_dataset.zip (MD5 ACAF57DEC5B52F0254E6F42C64C4314F verified).\n"
        "- metadata.csv, readme.md: unchanged copies from dataset.zip (metadata DATE is year-month only).\n"
        "- protocols/<P>_protokoll.txt: the acquisition protocol of each patient folder (ISIS MER log: sites, depths, channel\n"
        "  configuration, impedances, hardware filters, stimulation settings). Changed for privacy only: dates 'YYYY-MM-DD' -> day 01,\n"
        "  and the numeric archive folder/file ids in the .dat paths -> '<id>'. Times of day are kept.\n"
        "- codes.zip: the authors' codes.zip (parser, signal-quality classifier with its training segments and trained model,\n"
        "  brain-layer classifier, audio converter). Repacked: member dates day 01, compiled __pycache__/*.pyc dropped, and one\n"
        "  first name in the parser's folder-exclusion list replaced by [REDACTED]. Same code on GitHub:\n"
        "  https://github.com/kasiaOsowska/Deep-Brain-Stimulation-Microelectrode-Recordings\n"
        "- file_map.tsv: every source CSV -> BIDS run, with its columns, size and SHA-256 (of the released CSV).\n"
        "- Not included: the signal CSVs themselves (their values are in the BIDS runs, float32), display_signals.exe (Windows viewer).\n")
    # ---- code, metadata ----
    os.makedirs(os.path.join(OUT, "code"), exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    for x in ("f2_gdansk_convert.py", "b3w10_common.py", "b3w3_common.py"):
        shutil.copy(os.path.join(here, x), os.path.join(OUT, "code", x))
    nmain = sum(1 for r in recs if "_task-mer_" in r["stem"])
    nann = len(recs) - nmain
    hrs = sum(r["dur"] for r in recs) / 3600
    hrs_main = sum(r["dur"] for r in recs if "_task-mer_" in r["stem"]) / 3600
    durs = np.array([r["dur"] for r in recs if "_task-mer_" in r["stem"]])
    summ = {"subjects": len(pats), "runs": len(recs), "runs_mer": nmain, "runs_merannot": nann, "hours": hrs, "hours_mer": hrs_main,
            "dur_median_s": float(np.median(durs)) if len(durs) else None, "dur_max_s": float(durs.max()) if len(durs) else None,
            "dur_min_s": float(durs.min()) if len(durs) else None,
            "rt_all": all(r["roundtrip_ok"] for r in recs), "n_rt_fail": sum(not r["roundtrip_ok"] for r in recs),
            "contiguous_all": all(r["contiguous"] for r in recs), "n_noncontiguous": sum(not r["contiguous"] for r in recs),
            "max_abs_err_vs_csv": max(r["max_abs_err_vs_csv_float64"] for r in recs),
            "exact32_all": all(r["exact_float32"] for r in recs), "raw_exact_all": all(r["raw_exact"] for r in recs),
            "runs_with_raw": sum(r["has_raw"] for r in recs), "clip_ok_all": all(r["clip_ok"] for r in recs),
            "columns": Counter("|".join(r["cols"]) for r in recs), "no_site": sum(r["site_no"] is None for r in recs),
            "protocol_scrub": {"dates": sum(c["dates"] for c in prot.values()), "dat_ids": sum(c["dat_ids"] for c in prot.values()), "files": len(prot)},
            "codes_zip": zred}
    json.dump({"summary": summ, "notes": notes, "recordings": recs}, open(os.path.join(OUT, "code", "conversion_report.json"), "w"), indent=1, default=str)
    AUTH = ["Katarzyna Osowska", "Julian Szymański", "Witold Libionka"]
    wjson(os.path.join(OUT, "dataset_description.json"), {
        "Name": "DBS microelectrode recordings from 132 patients, University Clinical Centre Gdańsk (Osowska, Szymański, Libionka 2026; derivative: authors' band-passed, classifier-selected MER segments)",
        "BIDSVersion": "1.10.0", "DatasetType": "derivative", "License": "CC0-1.0", "Authors": AUTH,
        "Acknowledgements": "Recordings from DBS procedures at the University Clinical Centre in Gdańsk, Poland; data and code released on MOST Wiedzy (Gdańsk University of Technology) by J. Szymański et al. Repackaged to BIDS for NEMAR.",
        "HowToAcknowledge": "Cite Osowska K., Szymański J., Libionka W. (2026) A Dataset of Microelectrode Recordings from Deep Brain Stimulation Procedures. Scientific Data 13:870, doi:10.1038/s41597-026-07492-w, and the MOST Wiedzy dataset doi:10.34808/nq8v-t162.",
        "Funding": ["Gdańsk University of Technology (partial)", "Cloud Artificial Intelligence Service Engineering (CAISE) platform, grant KPOD.05.10-IW.10-0005/24, European IPCEI-CIS programme, financed by the National Recovery and Resilience Plan (NRRP)"],
        "EthicsApprovals": ["Ethics Commission of Medical Sciences Department, University of Warmia and Mazury, 43/2016", "Bioethical Commission of Nicolaus Copernicus University in Toruń, KB 416/2008 and KB 28/2010 (extension 2021)"],
        "ReferencesAndLinks": ["https://doi.org/10.34808/nq8v-t162", "https://doi.org/10.1038/s41597-026-07492-w", "https://github.com/kasiaOsowska/Deep-Brain-Stimulation-Microelectrode-Recordings"],
        "Keywords": ["deep brain stimulation", "microelectrode recording", "MER", "basal ganglia", "globus pallidus", "subthalamic nucleus", "Parkinson's disease", "Huntington's disease", "dystonia", "intraoperative"],
        "SourceDatasets": [{"DOI": f"doi:{DOI}", "URL": "https://mostwiedzy.pl/en/open-research-data/dbs-recordings,1209080239123979-0", "Version": "DBS_dataset_v3 (MOST Wiedzy record 0_12090802391239790_ORD, published 2025)"}],
        "GeneratedBy": [
            {"Name": "Authors' processing (codes/parser + signal_quality_classifier)", "Description": "ISIS MER .dat files (int16, 20 kHz; not released) decoded, overflow-corrected, band-passed 500-5000 Hz (4th-order Butterworth, causal lfilter) and clipped to +-500 instrument units; a Random Forest signal-quality classifier on 0.5 s windows then kept, per recording, the longest continuous fragment with >= 3 consecutive 'Neuronal activity' windows (>= 1.5 s) or else the longest artefact-free fragment >= 2 s; an electrophysiologist annotated 29 excerpts (striatum/putamen, GPe, GPi).",
             "CodeURL": "https://github.com/kasiaOsowska/Deep-Brain-Stimulation-Microelectrode-Recordings"},
            {"Name": "f2_gdansk_convert.py", "Description": "Repackaging of every released CSV as one BrainVision IEEE float32 run (values bit-exact to float32 of the CSV values; integer '1: raw' values exact), electrode label as acq, protocol order as run; depth, source file and segment offset in scans.tsv and events.tsv; annotated excerpts as task-merannot runs and as events in the matching task-mer runs. Code in code/.", "Version": "2026-10-08"}],
    })
    mx = summ["max_abs_err_vs_csv"]
    open(os.path.join(OUT, "README"), "w").write(f"""# DBS microelectrode recordings, University Clinical Centre Gdańsk (Osowska, Szymański, Libionka 2026) - DERIVATIVE

Intraoperative microelectrode recordings (MER) from {len(pats)} patients undergoing deep brain stimulation (DBS) surgery at the
University Clinical Centre in Gdańsk, Poland (Parkinson's disease, Huntington's disease, dystonia, tremor/other movement
disorders, epilepsy; awake surgery under local anaesthesia). Up to five parallel microelectrodes (Ben-gun: central, anterior,
posterior, medial, lateral; inomed ISIS MER system, MicroMacro electrodes) were advanced towards the target, recording at
successive depths relative to the MRI-planned target (typically from -10 mm; 0 mm = target). Scalp reference, chest ground.

## THIS IS A DERIVATIVE DATASET of short, filtered, classifier-selected segments
- The original surgical recordings (ISIS .dat files, int16, 20 kHz) were NOT released by the authors.
- Every run is ONE SHORT SEGMENT, not a continuous recording: from each depth recording the authors' Random Forest
  signal-quality classifier (0.5 s windows: neuronal activity / background / artifact) kept only the longest continuous
  fragment with >= 3 consecutive 'neuronal activity' windows, or else the longest artefact-free fragment of >= 2 s.
  Here: median {summ['dur_median_s']:.2f} s, range {summ['dur_min_s']:.2f}-{summ['dur_max_s']:.2f} s; {hrs_main:.2f} h in {nmain} runs.
  RecordingType is 'epoched' (one epoch per run). `source_time_start_s` (scans.tsv) is the offset of the kept segment
  within the unreleased site recording. Segments of different electrodes at the same depth were selected independently and
  are NOT time-aligned; each electrode/depth is its own run.
- The signal channel `<label>` is the CSV column '2: preprocessed': software band-pass 500-5000 Hz (4th-order Butterworth,
  causal scipy lfilter, after an int16 overflow correction) and amplitude clipped to +-500 (authors' parser, codes.zip).
- Units: the authors had no manufacturer conversion; amplitudes are 'instrument units' that they interpret as
  'µV-equivalent', possibly off by an unknown global factor (article, Methods). They are labelled µV here as in the
  release README; treat the absolute scale with that caveat.
- Where the CSV also has '1: raw' (the {summ['runs_with_raw']} annotated excerpts{'' if summ['runs_with_raw'] == nann else ' and some others'}), it is kept as a second channel
  `<label>_raw`: the int16 values decoded from the .dat file over the same samples, unfiltered (also instrument units).

## Layout
- `sub-<P###>`: the authors' patient code (P001-P132). `participants.tsv`: age, sex, ICD-10 diagnosis, surgical target and
  procedure year-month from metadata.csv (36 patients have no retrievable diagnosis/target: n/a).
- `task-mer_acq-<label>_run-<k>`: one released MER/<P>/depth<d>_kanal<label>.csv. acq = electrode label from the file name
  in lower case (central, anterior, posterior, medial, lateral; a leading digit 3-5 is the amplifier channel number that the
  authors' parser kept, e.g. 3central; `centrallateral` is the label 'Central3Lateral' that comes from a protocol line
  '2: Central3: Lateral' and cannot be resolved to one electrode). Run k follows the protocol site order, then depth.
  Side (hemisphere) is not recorded in the release: n/a.
- `task-merannot_acq-<label>_run-<k>`: the {nann} expert-annotated excerpts (brain_layer_annotations; patients P106 and P121,
  GPi-targeted trajectories): striatum/putamen, external (GPe) or internal (GPi) globus pallidus. The structure is in
  scans.tsv (`annotated_structure`) and events.tsv. When the excerpt's interval on the source time axis lies inside the
  matching task-mer segment (same patient, electrode, depth), that run's events.tsv carries an `annotated_<structure>`
  interval too. This mapping uses the CSV 'Time' values only: the excerpts' preprocessed values are NOT sample-identical to
  the task-mer files (the authors evidently filtered the excerpts separately), see code/conversion_report.json.
- `sub-<P>_scans.tsv`: depth (mm from target), electrode label, trajectory, protocol site number, source CSV, segment
  start/end on the source time axis, samples, raw-channel flag, annotated structure.
- `events.tsv` per run: the whole segment (`mer_segment`, with depth) plus any annotated intervals.
- Data format: BrainVision IEEE float32 at 20 kHz. The CSV text values (float64 printouts of float32 processing) were
  cast to float32; max |difference| {mx:.3g}; every .eeg was read back and is bit-identical to float32 of the CSV values;
  the integer raw values are exact. MNE reads every file.
- No electrode coordinates are released; brain location is given by depth relative to the planned target, the surgical
  target per patient and the expert structure labels.

## sourcedata/most_wiedzy_nq8v-t162
metadata.csv and readme.md (unchanged), the per-patient acquisition protocols (dates set to day 01, archive ids in .dat
paths replaced by <id>), the authors' codes.zip (member dates day 01, .pyc dropped, one first name in the parser's
exclusion list redacted), and file_map.tsv (every source CSV -> BIDS run, with SHA-256 of the released CSV).

## Related data (not duplicates)
Mendeley Data tr93krswn2 ('RAW Deep Brain Stimulation Recordings', 46 PD surgeries) and h9tbvgjgjv ('STN/SNr raw DBS
recordings', 83 PD surgeries) by K.A. Ciecierski and T. Mandat (CC BY-NC 3.0, Warsaw) are separate MER collections; they are
not on NEMAR. This Gdańsk release states a single centre (University Clinical Centre in Gdańsk) and shares no patient codes
or file names with them; no patient overlap can be established from either release.

## Source
- Dataset: Szymański J. et al., "DBS recordings", MOST Wiedzy, Gdańsk University of Technology, doi:10.34808/nq8v-t162 (CC0 1.0).
- Article: Osowska K., Szymański J., Libionka W. (2026) "A Dataset of Microelectrode Recordings from Deep Brain Stimulation
  Procedures", Scientific Data 13:870, doi:10.1038/s41597-026-07492-w.
- Code: https://github.com/kasiaOsowska/Deep-Brain-Stimulation-Microelectrode-Recordings
- Funding: Gdańsk University of Technology (partial); CAISE platform grant KPOD.05.10-IW.10-0005/24 (European IPCEI-CIS, NRRP).
- Ethics: University of Warmia and Mazury 43/2016; Nicolaus Copernicus University Toruń KB 416/2008, KB 28/2010.

## Privacy
Patient codes P001-P132 are the authors' pseudonyms. metadata.csv holds year-month only. The protocols carried full
procedure dates (day set to 01 here) and numeric archive ids in file paths (replaced). No names were found in file names,
CSV headers or protocols.
""")
    open(os.path.join(OUT, "CHANGES"), "w").write("1.0.0 2026-10-08\n  - Derivative repackaging of the MOST Wiedzy DBS MER release doi:10.34808/nq8v-t162 for NEMAR.\n")
    open(os.path.join(OUT, ".bidsignore"), "w").write("code/\n")
    print(json.dumps(summ, default=str))
    print("notes", len(notes))
    for n in notes[:80]:
        print("NOTE", n)
    print("CONVERT_DONE")


if __name__ == "__main__":
    main()
