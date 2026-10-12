# Original release files added to sourcedata/ (2026-10)

Added in the sourcedata completion pass of 2026-10: 6675 files, 47.55 GB.

- The signal CSVs of the release archive dataset.zip, unchanged: 6646 MER recordings dataset/MER/P###/depth-*_kanal*.csv and 29 CSVs under dataset/brain_layer_annotations/ (paths as in file_map.tsv; 6675 files size-checked against it).
- Not added: the 132 protocol files (already here as protocols/<P>_protokoll.txt, dates set to day 01), display_signals.exe (Windows signal viewer, no data), metadata.csv and readme.md (already here).

Changes for privacy:

- none (files as downloaded).

Before upload every added file passed the campaign privacy gate (file names, text, MAT headers and strings, zip member names and text members: no day-of-month other than 01, no names or identifiers). The CSVs hold only a header line and numbers (checked byte-wise); the header lines were scanned.
