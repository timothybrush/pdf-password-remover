#!/usr/bin/env python3
"""
PDF Password Remover

A command-line tool to recursively remove passwords from PDF files.
Processes all PDFs in an input directory and saves decrypted copies
to an output directory while preserving the folder structure.

Usage:
    uv run pdf_password_remover.py --input ./encrypted --output ./decrypted
    uv run pdf_password_remover.py -i ./pdfs -o ./output --allow-empty-password
    uv run pdf_password_remover.py -i ./pdfs -o ./output --password "secret"

Requirements (install with uv):
    uv add typer rich pydantic 'pypdf[crypto]'
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader, PdfWriter
from pypdf.errors import FileNotDecryptedError, PdfReadError
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.prompt import Prompt

# Initialize Rich console for styled output
console = Console()

# Create Typer app
app = typer.Typer(
    name="pdf-password-remover",
    help="Remove passwords from PDF files in a directory.",
    no_args_is_help=True,
    add_completion=False,
)


# =============================================================================
# Data Models
# =============================================================================


class Config(BaseModel):
    """Validated configuration for PDF processing."""

    input_dir: Path = Field(..., description="Directory containing encrypted PDFs")
    output_dir: Path = Field(..., description="Directory for decrypted PDFs")
    allow_empty_password: bool = Field(
        default=False,
        description="Try empty password before prompting",
    )
    password: str | None = Field(
        default=None,
        description="Password to try for all PDFs",
    )
    force: bool = Field(
        default=False,
        description="Overwrite existing output files",
    )

    @field_validator("input_dir")
    @classmethod
    def validate_input_dir(cls, path: Path) -> Path:
        """Ensure input directory exists and is a directory."""
        resolved = path.resolve()
        if not resolved.exists():
            raise ValueError(f"Input directory does not exist: {resolved}")
        if not resolved.is_dir():
            raise ValueError(f"Input path is not a directory: {resolved}")
        return resolved

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, path: Path) -> Path:
        """Ensure output path is not an existing file."""
        resolved = path.resolve()
        if resolved.exists() and not resolved.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {resolved}")
        return resolved


@dataclass
class ProcessingResult:
    """Result of processing a single PDF file."""

    source: Path
    destination: Path
    status: str  # "success", "skipped", "failed"
    message: str = ""


@dataclass
class Summary:
    """Aggregate results from processing all PDFs."""

    total: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    results: list[ProcessingResult] = field(default_factory=list)

    def add(self, result: ProcessingResult) -> None:
        """Add a result and update counters."""
        self.results.append(result)
        self.total += 1
        match result.status:
            case "success":
                self.succeeded += 1
            case "skipped":
                self.skipped += 1
            case "failed":
                self.failed += 1


# =============================================================================
# Core Functions
# =============================================================================


def find_pdfs(directory: Path) -> list[Path]:
    """
    Recursively find all PDF files in a directory.

    Args:
        directory: Root directory to search.

    Returns:
        Sorted list of PDF file paths.
    """
    pdfs = [p for p in directory.rglob("*.pdf") if p.is_file()]
    return sorted(pdfs, key=lambda p: str(p).casefold())


def write_pdf_atomic(writer: PdfWriter, destination: Path) -> None:
    """
    Write a PDF to disk atomically using a temp file.

    This prevents corrupted output files if the process is interrupted.

    Args:
        writer: PdfWriter with pages to write.
        destination: Final output path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory, then rename
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=destination.parent,
        suffix=".pdf",
    ) as tmp:
        writer.write(tmp)
        tmp_path = Path(tmp.name)

    # Atomic rename (same filesystem)
    tmp_path.replace(destination)


def try_decrypt(
    reader: PdfReader,
    *,
    allow_empty_password: bool,
    preset_password: str | None,
) -> tuple[bool, bool]:
    """
    Attempt to decrypt a PDF reader.

    Tries passwords in order:
    1. Empty password (if allowed)
    2. Preset password (if provided)

    Args:
        reader: PdfReader instance to decrypt.
        allow_empty_password: Whether to try empty password first.
        preset_password: Password to try before prompting.

    Returns:
        Tuple of (decrypted_successfully, needs_interactive_prompt).
    """
    if not reader.is_encrypted:
        return True, False

    # Try empty password
    if allow_empty_password:
        try:
            if reader.decrypt(""):
                return True, False
        except Exception:
            pass

    # Try preset password
    if preset_password is not None:
        try:
            if reader.decrypt(preset_password):
                return True, False
        except Exception:
            pass

    return False, True


def decrypt_interactively(reader: PdfReader, filename: str, max_attempts: int = 3) -> bool:
    """
    Prompt user for password and attempt decryption.

    Args:
        reader: PdfReader instance to decrypt.
        filename: Name of file (for display).
        max_attempts: Maximum password attempts allowed.

    Returns:
        True if decryption succeeded, False otherwise.
    """
    for attempt in range(1, max_attempts + 1):
        password = Prompt.ask(
            f"[yellow]Password for[/yellow] [cyan]{filename}[/cyan] "
            f"[dim](attempt {attempt}/{max_attempts})[/dim]",
            password=True,
        )
        try:
            if reader.decrypt(password):
                return True
        except Exception:
            pass
        console.print("[red]Incorrect password[/red]")

    return False


def process_single_pdf(
    source: Path,
    destination: Path,
    *,
    allow_empty_password: bool,
    preset_password: str | None,
    force: bool,
) -> ProcessingResult:
    """
    Process a single PDF: decrypt if needed and save without password.

    Args:
        source: Path to input PDF.
        destination: Path for output PDF.
        allow_empty_password: Try empty password first.
        preset_password: Password to try for all files.
        force: Overwrite existing output files.

    Returns:
        ProcessingResult with status and details.
    """
    # Skip if output exists and not forcing overwrite
    if destination.exists() and not force:
        return ProcessingResult(
            source=source,
            destination=destination,
            status="skipped",
            message="Output file exists (use --force to overwrite)",
        )

    try:
        with source.open("rb") as f:
            reader = PdfReader(f)

            # Handle encryption
            if reader.is_encrypted:
                decrypted, needs_prompt = try_decrypt(
                    reader,
                    allow_empty_password=allow_empty_password,
                    preset_password=preset_password,
                )

                if not decrypted and needs_prompt:
                    decrypted = decrypt_interactively(reader, source.name)

                if not decrypted:
                    return ProcessingResult(
                        source=source,
                        destination=destination,
                        status="failed",
                        message="Decryption failed (incorrect password)",
                    )

            # Create writer and copy all pages
            writer = PdfWriter()
            for page in reader.pages:
                writer.add_page(page)

            # Preserve metadata if available
            if reader.metadata:
                try:
                    metadata = {
                        k: str(v) for k, v in reader.metadata.items() if v is not None
                    }
                    writer.add_metadata(metadata)
                except Exception:
                    pass  # Metadata preservation is best-effort

            # Write output atomically
            write_pdf_atomic(writer, destination)

            return ProcessingResult(
                source=source,
                destination=destination,
                status="success",
                message="Decrypted" if reader.is_encrypted else "Copied (not encrypted)",
            )

    except FileNotFoundError:
        return ProcessingResult(
            source=source,
            destination=destination,
            status="failed",
            message="File not found",
        )
    except PdfReadError as e:
        return ProcessingResult(
            source=source,
            destination=destination,
            status="failed",
            message=f"Invalid PDF: {e}",
        )
    except PermissionError as e:
        return ProcessingResult(
            source=source,
            destination=destination,
            status="failed",
            message=f"Permission denied: {e}",
        )
    except FileNotDecryptedError:
        return ProcessingResult(
            source=source,
            destination=destination,
            status="failed",
            message="File remains encrypted after decryption attempt",
        )
    except Exception as e:
        return ProcessingResult(
            source=source,
            destination=destination,
            status="failed",
            message=f"Unexpected error: {e}",
        )


def process_directory(config: Config) -> Summary:
    """
    Process all PDFs in the input directory.

    Args:
        config: Validated processing configuration.

    Returns:
        Summary with counts and individual results.
    """
    summary = Summary()
    pdfs = find_pdfs(config.input_dir)

    if not pdfs:
        console.print("[yellow]No PDF files found in input directory.[/yellow]")
        return summary

    console.print(f"\n[cyan]Found {len(pdfs)} PDF file(s) to process[/cyan]\n")

    # Process with progress bar
    with Progress(
        TextColumn("[bold blue]Processing"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.description}[/dim]"),
        console=console,
    ) as progress:
        task = progress.add_task("", total=len(pdfs))

        for source in pdfs:
            # Calculate output path preserving directory structure
            relative_path = source.relative_to(config.input_dir)
            destination = config.output_dir / relative_path

            progress.update(task, description=source.name)

            result = process_single_pdf(
                source=source,
                destination=destination,
                allow_empty_password=config.allow_empty_password,
                preset_password=config.password,
                force=config.force,
            )

            summary.add(result)

            # Display result
            match result.status:
                case "success":
                    console.print(f"[green]✓[/green] {source.name}: {result.message}")
                case "skipped":
                    console.print(f"[yellow]○[/yellow] {source.name}: {result.message}")
                case "failed":
                    console.print(f"[red]✗[/red] {source.name}: {result.message}")

            progress.advance(task)

    return summary


def display_summary(summary: Summary, output_dir: Path) -> None:
    """Display a formatted summary panel."""
    status_color = "green" if summary.failed == 0 else "yellow"

    console.print(
        Panel(
            f"[green]Succeeded:[/green] {summary.succeeded}\n"
            f"[yellow]Skipped:[/yellow] {summary.skipped}\n"
            f"[red]Failed:[/red] {summary.failed}\n"
            f"[cyan]Output directory:[/cyan] {output_dir}",
            title="[bold]Summary[/bold]",
            border_style=status_color,
        )
    )


# =============================================================================
# CLI Command
# =============================================================================


@app.command()
def main(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Input directory containing PDF files.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for decrypted PDFs.",
        ),
    ],
    allow_empty_password: Annotated[
        bool,
        typer.Option(
            "--allow-empty-password",
            "-e",
            help="Try empty password before prompting.",
        ),
    ] = False,
    password: Annotated[
        str | None,
        typer.Option(
            "--password",
            "-p",
            help="Password to try for all encrypted PDFs.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing output files.",
        ),
    ] = False,
) -> None:
    """
    Remove passwords from all PDFs in a directory.

    Recursively processes PDF files in the input directory, removes
    password protection, and saves decrypted copies to the output
    directory while preserving the folder structure.
    """
    console.print(
        Panel(
            "[bold cyan]PDF Password Remover[/bold cyan]",
            border_style="cyan",
        )
    )

    # Validate configuration with Pydantic
    try:
        config = Config(
            input_dir=input_dir,
            output_dir=output_dir,
            allow_empty_password=allow_empty_password,
            password=password,
            force=force,
        )
    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(code=1)

    # Ensure input and output are different
    if config.input_dir == config.output_dir:
        console.print("[red]Error:[/red] Input and output directories must be different.")
        raise typer.Exit(code=1)

    # Create output directory
    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        console.print(
            f"[red]Error:[/red] Cannot create output directory: {config.output_dir}"
        )
        raise typer.Exit(code=1)

    # Process PDFs
    summary = process_directory(config)

    # Display summary
    if summary.total > 0:
        console.print()
        display_summary(summary, config.output_dir)

    # Exit with error code if any failures
    if summary.failed > 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
