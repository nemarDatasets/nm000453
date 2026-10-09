Files from the MOST Wiedzy release doi:10.34808/nq8v-t162 (CC0-1.0), archive DBS_dataset.zip (MD5 ACAF57DEC5B52F0254E6F42C64C4314F verified).
- metadata.csv, readme.md: unchanged copies from dataset.zip (metadata DATE is year-month only).
- protocols/<P>_protokoll.txt: the acquisition protocol of each patient folder (ISIS MER log: sites, depths, channel
  configuration, impedances, hardware filters, stimulation settings). Changed for privacy only: dates 'YYYY-MM-DD' -> day 01,
  and the numeric archive folder/file ids in the .dat paths -> '<id>'. Times of day are kept.
- codes.zip: the authors' codes.zip (parser, signal-quality classifier with its training segments and trained model,
  brain-layer classifier, audio converter). Repacked: member dates day 01, compiled __pycache__/*.pyc dropped, and one
  first name in the parser's folder-exclusion list replaced by [REDACTED]. Same code on GitHub:
  https://github.com/kasiaOsowska/Deep-Brain-Stimulation-Microelectrode-Recordings
- file_map.tsv: every source CSV -> BIDS run, with its columns, size and SHA-256 (of the released CSV).
- Not included: the signal CSVs themselves (their values are in the BIDS runs, float32), display_signals.exe (Windows viewer).
