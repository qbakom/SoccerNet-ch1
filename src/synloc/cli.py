"""CLI entry point for SoccerNet SynLoc pipeline."""

import logging

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@click.group()
def cli():
    """SoccerNet SynLoc 2026 Challenge CLI."""
    pass


# --- Data commands ---


@cli.group()
def data():
    """Dataset download and validation."""
    pass


@data.command()
@click.option("--version", default="fullhd", help="Image version: 'fullhd' or default 4K")
@click.option("--splits", default="train,valid,test,challenge", help="Comma-separated splits")
def download(version, splits):
    """Download the SoccerNet SynLoc dataset."""
    from synloc.dataset import download as _download

    split_list = tuple(s.strip() for s in splits.split(","))
    _download(version=version, splits=split_list)
    click.echo("Download complete.")


@data.command()
def dummy():
    """Create a small dummy dataset for pipeline testing."""
    from synloc.dataset import create_dummy_dataset

    path = create_dummy_dataset()
    click.echo(f"Dummy dataset created at {path}")


@data.command()
def validate():
    """Validate dataset integrity."""
    from synloc.dataset import validate as _validate

    stats = _validate()
    for split, info in stats.items():
        if info.get("exists"):
            click.echo(f"  {split}: {info['images']} images, {info['annotations']} annotations")
        else:
            click.echo(f"  {split}: MISSING")


# --- Training commands ---


@cli.command()
@click.option("--model", default="m", type=click.Choice(["tiny", "s", "m", "l"]), help="YOLOX variant")
@click.option("--resolution", default=960, type=click.Choice([640, 960, 1280], case_sensitive=False), help="Input resolution")
@click.option("--gpus", default=1, type=int, help="Number of GPUs")
@click.option("--epochs", default=None, type=int, help="Override default 300 epochs")
@click.option("--resume", default=None, type=str, help="Checkpoint to resume from")
@click.option("--no-amp", is_flag=True, help="Disable automatic mixed precision")
def train(model, resolution, gpus, epochs, resume, no_amp):
    """Train YOLOX-Pose model."""
    from synloc.modeling.train import train as _train

    _train(
        model_size=model,
        resolution=int(resolution),
        num_gpus=gpus,
        epochs=epochs,
        resume_from=resume,
        amp=not no_amp,
    )


# --- Evaluation commands ---


@cli.group()
def eval():
    """Evaluation and submission."""
    pass


@eval.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--model", default="m", type=click.Choice(["tiny", "s", "m", "l"]))
@click.option("--resolution", default=960, type=click.Choice([640, 960, 1280], case_sensitive=False))
@click.option("--split", default="valid", type=click.Choice(["valid", "test"]))
def run(checkpoint, model, resolution, split):
    """Evaluate a checkpoint on a dataset split."""
    from synloc.modeling.predict import evaluate

    metrics_file = evaluate(
        checkpoint=checkpoint,
        model_size=model,
        resolution=int(resolution),
        split=split,
    )
    click.echo(f"Metrics: {metrics_file}")


@eval.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--model", default="m", type=click.Choice(["tiny", "s", "m", "l"]))
@click.option("--resolution", default=960, type=click.Choice([640, 960, 1280], case_sensitive=False))
def submit(checkpoint, model, resolution):
    """Generate challenge submission zip."""
    from synloc.modeling.predict import submit as _submit

    result = _submit(
        checkpoint=checkpoint,
        model_size=model,
        resolution=int(resolution),
    )
    click.echo(f"Submission: {result}")


# --- Visualization commands ---


@cli.command()
@click.option("--split", default="train", help="Dataset split to visualize")
@click.option("--max-images", default=5, type=int, help="Number of images to plot")
def visualize(split, max_images):
    """Generate pitch position visualizations."""
    from synloc.visualization import plot_positions

    plot_positions(split=split, max_images=max_images)
    click.echo("Visualizations saved to reports/figures/")


if __name__ == "__main__":
    cli()
