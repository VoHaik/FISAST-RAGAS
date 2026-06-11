from pathlib import Path

import typer
from dotenv import load_dotenv

from src.evaluation_dataset.chunker import chunk_pdf_pages
from src.evaluation_dataset.config import EvaluationDatasetConfig
from src.evaluation_dataset.document_loader import load_input_directory
from src.evaluation_dataset.exporter import export_dataset
from src.evaluation_dataset.generator import generate_ragas_testset
from src.evaluation_dataset.pdf_loader import PdfExtractionQualityError


app = typer.Typer(no_args_is_help=True)


def _resolve_input_dir(input_dir: Path | None, input_pdf_dir: Path | None) -> Path:
    if input_dir is not None:
        return input_dir
    if input_pdf_dir is not None:
        return input_pdf_dir
    raise typer.BadParameter(
        "Provide --input-dir for .pdf/.md input, or --input-pdf-dir for legacy PDF-only usage"
    )


@app.command()
def generate(
    input_dir: Path | None = typer.Option(None, exists=True, file_okay=False),
    input_pdf_dir: Path | None = typer.Option(None, exists=True, file_okay=False),
    output_path: Path = typer.Option(...),
    output_format: str = typer.Option("jsonl"),
    chunk_size: int = typer.Option(1000),
    chunk_overlap: int = typer.Option(150),
    testset_size: int = typer.Option(100),
    single_hop_specific_ratio: float = typer.Option(0.5),
    multi_hop_specific_ratio: float = typer.Option(0.25),
    multi_hop_abstract_ratio: float = typer.Option(0.25),
    provider: str | None = typer.Option(None),
    llm_base_url: str | None = typer.Option(None),
    embeddings_base_url: str | None = typer.Option(None),
    generator_model: str | None = typer.Option(None),
    embeddings_model: str | None = typer.Option(None),
    temperature: float | None = typer.Option(None),
    timeout: int | None = typer.Option(None),
    num_predict: int | None = typer.Option(None),
    llm_format: str | None = typer.Option(None),
) -> None:
    load_dotenv()
    resolved_input_dir = _resolve_input_dir(input_dir, input_pdf_dir)
    config = EvaluationDatasetConfig.from_env(
        input_pdf_dir=resolved_input_dir,
        output_path=output_path,
        output_format=output_format,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        testset_size=testset_size,
        single_hop_specific_ratio=single_hop_specific_ratio,
        multi_hop_specific_ratio=multi_hop_specific_ratio,
        multi_hop_abstract_ratio=multi_hop_abstract_ratio,
        provider=provider,
        generator_model=generator_model,
        embeddings_model=embeddings_model,
        llm_base_url=llm_base_url,
        embeddings_base_url=embeddings_base_url,
        temperature=temperature,
        timeout=timeout,
        num_predict=num_predict,
        llm_format=llm_format,
    )
    config.validate()

    try:
        source_documents = load_input_directory(config.input_pdf_dir)
    except PdfExtractionQualityError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    documents = chunk_pdf_pages(
        source_documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    frame = generate_ragas_testset(documents, config)
    export_dataset(frame, config.output_path, config.output_format)

    typer.echo(f"Generated {len(frame)} rows at {config.output_path}")


if __name__ == "__main__":
    app()
