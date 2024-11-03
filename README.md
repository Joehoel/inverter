# PowerPoint Color Inverter

A Python utility that inverts colors in PowerPoint presentations. This tool can process individual PPTX files or entire directories, inverting all colors including slide backgrounds, text, and images.

## Features

- Invert colors in single PPTX files or entire directories
- Recursive directory processing
- Parallel processing for improved performance
- Progress bar with estimated time remaining
- Graceful interrupt handling
- Skip already processed files
- Maintains original directory structure
- Timeout protection for large files

## Requirements

- Python 3.7+
- uv (for dependency management)

## Installation

1. Install uv using the following command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a new virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

## Usage

### Basic Usage

Process a single file:

```bash
uv run cli.py presentation.pptx
```

Process a directory:

```bash
uv run cli.py path/to/presentations/
```

### Advanced Options

```bash
uv run cli.py [-h] [-o OUTPUT] [-w WORKERS] [-r] [-t TIMEOUT] [-f] input
```

Arguments:

- `input`: Input file or directory path
- `-o, --output`: Output file or directory path (optional)
- `-w, --workers`: Number of worker threads (default: CPU count \* 4)
- `-r, --recursive`: Process subdirectories recursively
- `-t, --timeout`: Timeout in seconds for processing each file (default: 300)
- `-f, --force`: Force processing of all files, even if they exist in output

### Examples

Process a single file with custom output path:

```bash
uv run cli.py presentation.pptx -o inverted.pptx
```

Process a directory recursively:

```bash
uv run cli.py presentations/ -r -o output/
```

Process with custom worker count and timeout:

```bash
uv run cli.py presentations/ -w 8 -t 600
```

## Output Structure

- When processing a single file without specifying an output path, the inverted file will be saved with " (inverted)" appended to the filename
- When processing a directory without specifying an output path, a new directory will be created with " (inverted)" appended to the original directory name
- The tool maintains the original directory structure when processing recursively
