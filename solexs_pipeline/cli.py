"""Console script for solexs_pipeline."""
import sys
import click
from .L0_interm import intermediate_directory
from .interm_L1 import L1_directory
from .logging import setup_logger, create_fileHandler
import os

log = setup_logger('solexs_pipeline')

@click.command()
@click.option('-i', '--input_file', help='Input SoLEXS Instrument Data File', multiple=True)
@click.option('-o', '--output_dir', default=None, show_default=True, help='Output Directory')
@click.option('-dt', '--data_type', default='L0', show_default=True, help='Raw/SP/L0')
@click.option('-sdd', '--SDD', default='12', show_default=True, help='1/2/12')
def main(input_file,output_dir,data_type,sdd,args=None):
    """Console script for solexs_pipeline."""
    click.echo("SoLEXS pipeline command line interface "
               "solexs_pipeline.cli.main")
    # click.echo("See click documentation at https://click.palletsprojects.com/")

    # fh = create_fileHandler(f'{os.path.basename(input_file)}.log')
    # log.addHandler(fh)
    log.info('SoLEXS pipeline command line interface initiated.')
    log.info(f'Input file: {input_file}')
    log.info(f'Output directory: {output_dir}')
    log.info(f'Data type: {data_type}')

    log.info('Initiating L0 to intermediate data pipeline.')
    interm_dir_paths = []
    for i_f in input_file:
        filename = i_f#click.option('filename',type=click.Path(exists=True))

        # Intermediate Directory
        interm_dir = intermediate_directory(filename, input_file_data_type=data_type)
        interm_date = interm_dir.solexs_bd.pld_header_SDD1.pld_utc_datetime[0].strftime('%Y%m%d')
        interm_output_dir = os.path.join(output_dir,'AL1_SOLEXS_'+ interm_date,'intermediate')
        interm_dir.make_interm_dir(output_dir=interm_output_dir)
        if sdd=='12':
            interm_dir.write_interm_files(SDD_number=1)
            interm_dir.write_interm_files(SDD_number=2)
        elif sdd=='1':
            interm_dir.write_interm_files(SDD_number=1)
        elif sdd=='2':
            interm_dir.write_interm_files(SDD_number=2)

        interm_dir_paths.append(interm_dir.output_dir)

    # L1 Directory
    log.info('Initiating intermediate to L1 data pipeline.')
    # interm_dir_path = interm_dir.output_dir
    l1_dir = L1_directory(interm_dir_paths)
    if sdd=='12':
        l1_pi_file_sdd1, l1_lc_file_sdd1 = l1_dir.create_l1_files(SDD_number=1)
        l1_dir.make_l1_dir(output_dir)
        l1_dir.write_l1_files(1, l1_pi_file_sdd1, l1_lc_file_sdd1)

        l1_pi_file_sdd2, l1_lc_file_sdd2 = l1_dir.create_l1_files(SDD_number=2)
        l1_dir.write_l1_files(2, l1_pi_file_sdd2, l1_lc_file_sdd2)
    
    if sdd=='1':
        l1_pi_file_sdd1, l1_lc_file_sdd1 = l1_dir.create_l1_files(SDD_number=1)
        l1_dir.make_l1_dir(output_dir)
        l1_dir.write_l1_files(1, l1_pi_file_sdd1, l1_lc_file_sdd1)

    if sdd=='2':
        l1_pi_file_sdd2, l1_lc_file_sdd2 = l1_dir.create_l1_files(SDD_number=2)
        l1_dir.make_l1_dir(output_dir)
        l1_dir.write_l1_files(2, l1_pi_file_sdd2, l1_lc_file_sdd2)


    # l1_dir.write_l1_files(SDD_number=1)
    # l1_dir.write_l1_files(SDD_number=2)

    log.info('SoLEXS pipeline executed successfully.')
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
