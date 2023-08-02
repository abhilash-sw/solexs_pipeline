"""Console script for solexs_pipeline."""
import sys
import click
from .L0_interm import intermediate_directory
from .interm_L1 import L1_directory

@click.command()
@click.option('-i','--input_file')
def main(input_file,args=None):
    """Console script for solexs_pipeline."""
    click.echo("Replace this message by putting your code into "
               "solexs_pipeline.cli.main11")
    click.echo("See click documentation at https://click.palletsprojects.com/")

    filename = input_file#click.option('filename',type=click.Path(exists=True))

    # Intermediate Directory
    interm_dir = intermediate_directory(filename)
    interm_dir.make_interm_dir(filename)
    interm_dir.write_interm_files(SDD_number=1)
    interm_dir.write_interm_files(SDD_number=2)

    # L1 Directory
    interm_dir_path = interm_dir.output_dir
    l1_dir = L1_directory(interm_dir_path)
    l1_dir.make_l1_dir()
    l1_dir.write_l1_files(SDD_number=1)
    l1_dir.write_l1_files(SDD_number=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
