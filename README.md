# PDF Password Remover

A modern CLI tool to recursively remove passwords from PDF files.

## Features

- **Recursive processing**: Processes all PDFs in a directory tree
- **Structure preservation**: Maintains folder hierarchy in output
- **Multiple password strategies**: Empty password, preset password, or interactive prompt
- **Atomic writes**: Safe file writing prevents corruption
- **Rich CLI**: Progress bars and colored output
- **Input validation**: Pydantic-validated configuration

## Installation

```bash
# Clone or download, then:
uv sync

# Or install dependencies directly:
uv add typer rich pydantic 'pypdf[crypto]'
```

## Usage

### Basic usage
```bash
uv run pdf_password_remover.py --input ./encrypted --output ./decrypted
```

### With a known password
```bash
uv run pdf_password_remover.py -i ./pdfs -o ./output --password "secret123"
```

### Try empty password first (common for view-only restrictions)
```bash
uv run pdf_password_remover.py -i ./pdfs -o ./output --allow-empty-password
```

### Force overwrite existing files
```bash
uv run pdf_password_remover.py -i ./pdfs -o ./output --force
```

### Combined options
```bash
uv run pdf_password_remover.py \
    --input ~/Documents/encrypted \
    --output ~/Documents/decrypted \
    --allow-empty-password \
    --password "backup-password" \
    --force
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Input directory containing PDF files (required) |
| `--output` | `-o` | Output directory for decrypted PDFs (required) |
| `--allow-empty-password` | `-e` | Try empty password before prompting |
| `--password` | `-p` | Password to try for all encrypted PDFs |
| `--force` | `-f` | Overwrite existing output files |
| `--help` | | Show help message |

## How It Works

1. **Discovery**: Recursively finds all `.pdf` files in the input directory
2. **Decryption**: For each encrypted PDF:
   - Tries empty password (if `--allow-empty-password`)
   - Tries preset password (if `--password` provided)
   - Prompts interactively (up to 3 attempts)
3. **Output**: Writes decrypted PDFs atomically to preserve data integrity
4. **Metadata**: Preserves PDF metadata when possible

## Requirements

- Python 3.12+
- typer
- rich
- pydantic
- pypdf[crypto]

## Exit Codes

- `0`: All files processed successfully
- `1`: One or more files failed to process
