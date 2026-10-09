# DBS dataset

This dataset contains processed microelectrode recordings from deep brain stimulation. The dataset's structure is designed to facilitate easy navigation and analysis. It includes CSV files and a signal visualization tool.

## Dataset Structure

- **`MER/`**: Folder containing processed MER. Processing includes:
  - Filtering signals using a Butterworth filter.
  - Limiting signal values to a range of -500 to 500 microvolts.
  - Extracting the best signal segments using a preselection classifier.
- **`brain_layer_annotations/`**: Folder containing brain structure annotations made by an expert.
- **`metadata.csv`**: Participant-level information

- **`display_signals.exe`**: Program for visualizing the data.

The dataset structure follows the pattern: Patient ID -> sequential recordings from the procedure. 
Each file is named using the following schema:

`depth<distance_from_reference_target_in_millimeters>_kanal<channel_name>.csv`

## CSV File Format

All files in the dataset are saved in CSV format with the following structure:

| Time (s) | Signal (µV) |
| -------- | ----------- |
| 0.001    | -123.45     |
| 0.002    | 456.78      |
| ...      | ...         |

## Visualization Tool

### Description

The dataset includes a program for visualizing signals. The program:

- Recursively searches for CSV files in a selected folder.
- Supports files containing the columns: `Time` and `2: preprocessed`.
- Generates a time vs. signal plot for the first file found.
- Extracts depth and channel information from the file name and displays it above the signal.

### Navigation Modes:

1. **Automatic Playback**: Displays consecutive plots every 2 seconds.
2. **Manual Scrolling**: Allows the user to manually switch between files.

Additionally:

- Single files can be loaded and visualized.
- The program supports zooming, panning, customizing the plot, and saving it as a PNG file.
