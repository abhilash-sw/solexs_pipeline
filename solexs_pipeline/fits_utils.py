#####################################################
# @Author: Abhilash Sarwade
# @Date:   2021-09-08 17:50:00
# @email: sarwade@ursc.gov.in
# @File Name: fits_utils.py
# @Project: solexs_pipeline

# @Last Modified time: 2023-07-31 02:58:01 pm
#####################################################

from builtins import str
from builtins import object
from astropy.io import fits
import numpy as np
import astropy.units as u
#import pkg_resources
import six
from typing import Any, Dict, List, Optional, Union
from .logging import setup_logger
import datetime

_NUMPY_TO_FITS_CODE = {
    # Integers
    np.int16: "I",
    np.int32: "J",
    np.int64: "K",
    np.uint16: "I",
    np.uint32: "J",
    # Floating point
    np.float32: "E",
    np.float64: "D",
    np.bool_: "L",  # https://docs.astropy.org/en/stable/io/fits/usage/table.html
    np.uint8: "B",
    np.uintc: "I",
}

log = setup_logger(f'solexs_pipeline.{__name__}')

class FITSFile(object):
    def __init__(self, primary_hdu=None, fits_extensions=None):

        hdu_list = []

        if primary_hdu is None:

            primary_hdu = fits.PrimaryHDU()

        else:

            assert isinstance(primary_hdu, fits.PrimaryHDU)

        hdu_list.append(primary_hdu)

        if fits_extensions is not None:

            fits_extensions = list(fits_extensions)

            hdu_list.extend([x.hdu for x in fits_extensions])

        # We embed instead of subclassing because the HDUList class has some weird interaction with the
        # __init__ and __new__ methods which makes difficult to do so (we couldn't figure it out)

        self._hdu_list = fits.HDUList(hdus=hdu_list)

    def writeto(self, *args, **kwargs):

        self._hdu_list.writeto(*args, **kwargs)

    # Update the docstring to be the same as the method we are wrapping

    writeto.__doc__ = fits.HDUList.writeto.__doc__

    def __getitem__(self, item):

        return self._hdu_list.__getitem__(item)

    def info(self, output=None):

        self._hdu_list.info(output)

    info.__doc__ = fits.HDUList.info.__doc__

    def index_of(self, key):

        return self._hdu_list.index_of(key)

    index_of.__doc__ = fits.HDUList.index_of.__doc__


class FITSExtension(object):

    # I use __new__ instead of __init__ because I need to use the classmethod .from_columns instead of the
    # constructor of fits.BinTableHDU

    def __init__(self, data_tuple, header_tuple):

        # Generate the header from the dictionary

        header = fits.Header(header_tuple)

        # Loop over the columns and generate them
        fits_columns = []

        for column_name, column_data in data_tuple:

            # Get type of column
            # NOTE: we assume the type is the same for the entire column

            test_value = column_data[0]

            # Generate FITS column

            # By default a column does not have units, unless the content is an astropy.Quantity

            units = None

            if isinstance(test_value, u.Quantity):

                # Probe the format

                try:

                    # Use the one already defined, if possible

                    format = _NUMPY_TO_FITS_CODE[column_data.dtype.type]

                except AttributeError:

                    # Try to infer it. Note that this could unwillingly upscale a float16 to a float32, for example

                    format = _NUMPY_TO_FITS_CODE[np.array(test_value.value).dtype.type]

                # check if this is a vector of quantities

                if test_value.shape:

                    format = "%i%s" % (test_value.shape[0], format)

                # Store the unit as text

                units = str(test_value.unit)

            elif isinstance(test_value, six.string_types):

                # Get maximum length, but make 1 as minimum length so if the column is completely made up of empty
                # string we still can work

                max_string_length = max(len(max(column_data, key=len)), 1)

                format = "%iA" % max_string_length

            elif np.isscalar(test_value):

                format = _NUMPY_TO_FITS_CODE[np.array(test_value).dtype.type]

            elif isinstance(test_value, list) or isinstance(test_value, np.ndarray):

                # Probably a column array
                # Check that we can convert it to a proper numpy type

                try:

                    # Get type of first number

                    col_type = np.array(test_value[0]).dtype.type

                except:

                    raise RuntimeError(
                        "Could not understand type of column %s" % column_name
                    )

                # Make sure we are not dealing with objects
                assert col_type != np.object and col_type != np.object_

                try:

                    _ = np.array(test_value, col_type)

                except:

                    raise RuntimeError(
                        "Column %s contain data which cannot be coerced to %s"
                        % (column_name, col_type)
                    )

                else:

                    # see if it is a string array

                    if test_value.dtype.type == np.string_:

                        max_string_length = max(column_data, key=len).dtype.itemsize

                        format = "%iA" % max_string_length

                    else:

                        # All good. Check the length
                        # NOTE: variable length arrays are not supported
                        line_length = len(test_value)
                        format = "%i%s" % (line_length, _NUMPY_TO_FITS_CODE[col_type])

            else:

                # Something we do not know

                raise RuntimeError(
                    "Column %s in dataframe contains objects which are not strings"
                    % column_name
                )

            this_column = fits.Column(
                name=column_name, format=format, unit=units, array=column_data
            )

            fits_columns.append(this_column)

        # Create the extension

        self._hdu = fits.BinTableHDU.from_columns(
            fits.ColDefs(fits_columns), header=header
        )

        # update the header to indicate that the file was created by 3ML
        self._hdu.header.set(
            "CREATOR",
            "solexs_pipeline",
            # "(Abhilash Sarwade, sarwade@ursc.gov.in)",
        )

    @property
    def hdu(self):

        return self._hdu

    @classmethod
    def from_fits_file_extension(cls, fits_extension):

        data = fits_extension.data

        data_tuple = []

        for name in data.columns.names:

            data_tuple.append((name, data[name]))

        header_tuple = list(fits_extension.header.items())

        return cls(data_tuple, header_tuple)


####################################################################################
# The following classes are used to create OGIP-compliant response files
# (at the moment only RMF are supported)


class EBOUNDS(FITSExtension):

    _HEADER_KEYWORDS = (
        ("EXTNAME", "EBOUNDS", "Extension name"),
        ("HDUCLASS", "OGIP    ", "format conforms to OGIP standard"),
        ("HDUVERS", "1.1.0   ", "Version of format (OGIP memo CAL/GEN/92-002a)"),
        (
            "HDUDOC",
            "OGIP memos CAL/GEN/92-002 & 92-002a",
            "Documents describing the forma",
        ),
        ("HDUVERS1", "1.0.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUVERS2", "1.1.0   ", "Obsolete - included for backwards compatibility"),
        ("CHANTYPE", "PI", "Channel type"),
        ("CONTENT", "OGIPResponse Matrix", "File content"),
        ("HDUCLAS1", "RESPONSE", "Extension contains response data  "),
        ("HDUCLAS2", "EBOUNDS ", "Extension contains EBOUNDS"),
        ("TLMIN1", 1, "Minimum legal channel number"),
    )

    def __init__(self, energy_boundaries):
        """
        Represents the EBOUNDS extension of a response matrix FITS file

        :param energy_boundaries: lower bound of channel energies (in keV)
        """

        n_channels = len(energy_boundaries)# - 1

        data_tuple = (
            ("CHANNEL", list(range(1, n_channels + 1))),
            ("E_MIN", energy_boundaries[:,0] * u.keV),
            ("E_MAX", energy_boundaries[:,1] * u.keV),
        )

        super(EBOUNDS, self).__init__(data_tuple, self._HEADER_KEYWORDS)


class MATRIX(FITSExtension):
    """
    Represents the MATRIX extension of a response FITS file following the OGIP format

    :param mc_energies_lo: lower bound of MC energies (in keV)
    :param mc_energies_hi: hi bound of MC energies (in keV)
    :param channel_energies_lo: lower bound of channel energies (in keV)
    :param channel_energies_hi: hi bound of channel energies (in keV
    :param matrix: the redistribution matrix, representing energy dispersion effects
    """

    _HEADER_KEYWORDS = [
        ("EXTNAME", "MATRIX", "Extension name"),
        ("HDUCLASS", "OGIP    ", "format conforms to OGIP standard"),
        ("HDUVERS", "1.1.0   ", "Version of format (OGIP memo CAL/GEN/92-002a)"),
        (
            "HDUDOC",
            "OGIP memos CAL/GEN/92-002 & 92-002a",
            "Documents describing the forma",
        ),
        ("HDUVERS1", "1.0.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUVERS2", "1.1.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUCLAS1", "RESPONSE", "dataset relates to spectral response"),
        ("HDUCLAS2", "RSP_MATRIX", "dataset is a spectral response matrix"),
        ("HDUCLAS3", "REDIST", "dataset represents energy dispersion only"),
        ("CHANTYPE", "PI ", "Detector Channel Type in use (PHA or PI)"),
        ("DETCHANS", None, "Number of channels"),
        ("FILTER", "", "Filter used"),
        ("TLMIN4", 1, "Minimum legal channel number"),
    ]

    def __init__(self, mc_energies, channel_energies, matrix):

        n_mc_channels = len(mc_energies)# - 1
        n_channels = len(channel_energies)# - 1

        assert matrix.shape == (
            n_channels,
            n_mc_channels,
        ), "Matrix has the wrong shape. Should be %i x %i, got %i x %i" % (
            n_channels,
            n_mc_channels,
            matrix.shape[0],
            matrix.shape[1],
        )

        ones = np.ones(n_mc_channels, np.int16)

        # We need to format the matrix as a list of n_mc_channels rows of n_channels length

        data_tuple = (
            ("ENERG_LO", mc_energies[:,0] * u.keV),
            ("ENERG_HI", mc_energies[:,1] * u.keV),
            ("N_GRP", ones),
            ("F_CHAN", ones),
            ("N_CHAN", np.ones(n_mc_channels, np.int16) * n_channels),
            ("MATRIX", matrix.T),
        )

        super(MATRIX, self).__init__(data_tuple, self._HEADER_KEYWORDS)

        # Update DETCHANS
        self.hdu.header.set("DETCHANS", n_channels)


class SPECRESP_MATRIX(MATRIX):
    """
    Represents the SPECRESP_MATRIX extension of a response FITS file following the OGIP format

    :param mc_energies_lo: lower bound of MC energies (in keV)
    :param mc_energies_hi: hi bound of MC energies (in keV)
    :param channel_energies_lo: lower bound of channel energies (in keV)
    :param channel_energies_hi: hi bound of channel energies (in keV
    :param matrix: the redistribution matrix, representing energy dispersion effects and effective area information
    """

    def __init__(self, mc_energies, channel_energies, matrix):

        # This is essentially exactly the same as MATRIX, but with a different extension name

        super(SPECRESP_MATRIX, self).__init__(
            mc_energies, channel_energies, matrix)

        # Change the extension name
        self.hdu.header.set("EXTNAME", "SPECRESP MATRIX")
        self.hdu.header.set("HDUCLAS3", "FULL")


class RMF(FITSFile):
    """
    A RMF file, the OGIP format for a matrix representing energy dispersion effects.

    """

    def __init__(self, mc_energies, ebounds, matrix, telescope_name, instrument_name):

        # Make sure that the provided iterables are of the right type for the FITS format

        mc_energies = np.array(mc_energies, np.float32)

        ebounds = np.array(ebounds, np.float32)

        # Create EBOUNDS extension
        ebounds_ext = EBOUNDS(ebounds)

        # Create MATRIX extension
        matrix_ext = MATRIX(mc_energies, ebounds, matrix)

        # Set telescope and instrument name
        matrix_ext.hdu.header.set("TELESCOP", telescope_name)
        matrix_ext.hdu.header.set("INSTRUME", instrument_name)

        # Create FITS file
        super(RMF, self).__init__(fits_extensions=[ebounds_ext, matrix_ext])


class RSP(FITSFile):
    """
    A response file, the OGIP format for a matrix representing both energy dispersion effects and effective area,
    in the same matrix.

    """

    def __init__(self, mc_energies, ebounds, matrix, telescope_name, instrument_name):

        # Make sure that the provided iterables are of the right type for the FITS format

        mc_energies = np.array(mc_energies, np.float32)

        ebounds = np.array(ebounds, np.float32)

        # Create EBOUNDS extension
        ebounds_ext = EBOUNDS(ebounds)

        # Create MATRIX extension
        matrix_ext = SPECRESP_MATRIX(mc_energies, ebounds, matrix)

        # Set telescope and instrument name
        matrix_ext.hdu.header.set("TELESCOP", telescope_name)
        matrix_ext.hdu.header.set("INSTRUME", instrument_name)

        # Create FITS file
        super(RSP, self).__init__(fits_extensions=[ebounds_ext, matrix_ext])



class SPECRESP(FITSExtension):

    _HEADER_KEYWORDS = (
        ("EXTNAME", "SPECRESP", "Extension name"),
        ("HDUCLASS", "OGIP    ", "format conforms to OGIP standard"),
        ("HDUVERS", "1.1.0   ", "Version of format (OGIP memo CAL/GEN/92-002a)"),
        (
            "HDUDOC",
            "OGIP memos CAL/GEN/92-002 & 92-002a",
            "Documents describing the forma",
        ),
        ("HDUVERS1", "1.0.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUVERS2", "1.1.0   ", "Obsolete - included for backwards compatibility"),
        ("CONTENT", "ARF", "File content"),
        ("HDUCLAS1", "RESPONSE", "Extension contains response data  "),
        ("HDUCLAS2", "SPECRESP ", "Extension contains Effective Area"),
        
    )

    def __init__(self,mc_energies, eff_area):
        """
        Represents the SPECRESP extension of a ARF FITS file

        :param mc_energies_lo: lower bound of MC energies (in keV)
        :param mc_energies_hi: hi bound of MC energies (in keV)
        :param eff_area: Effective area (in cm^2)
        """

        # n_channels = len(energy_boundaries)# - 1

        data_tuple = (
            ("ENERG_LO", mc_energies[:,0] * u.keV),
            ("ENERG_HI", mc_energies[:,1] * u.keV),
            ("SPECRESP", eff_area * u.cm * u.cm),
        )

        super(SPECRESP, self).__init__(data_tuple, self._HEADER_KEYWORDS)

class ARF(FITSFile):
    """
    A ARF file, the OGIP format for a array representing effective area.

    """

    def __init__(self, mc_energies, eff_area, telescope_name, instrument_name):

        # Make sure that the provided iterables are of the right type for the FITS format

        mc_energies = np.array(mc_energies, np.float32)

        eff_area = np.array(eff_area, np.float32)

        # Create EBOUNDS extension
        specresp_ext = SPECRESP(mc_energies,eff_area)

        # Create MATRIX extension
        

        # Set telescope and instrument name
        specresp_ext.hdu.header.set("TELESCOP", telescope_name)
        specresp_ext.hdu.header.set("INSTRUME", instrument_name)

        # Create FITS file
        super(ARF, self).__init__(fits_extensions=[specresp_ext])








def _atleast_2d_with_dtype(value, dtype=None):

    if dtype is not None:
        value = np.array(value, dtype=dtype)

    arr = np.atleast_2d(value)

    return arr


def _atleast_1d_with_dtype(value, dtype=None):

    if dtype is not None:
        value = np.array(value, dtype=dtype)

        if dtype == str:

            # convert None to NONE
            # which is needed for None Type args
            # to string arrays

            idx = np.core.defchararray.lower(value) == "none"

            value[idx] = "NONE"

    arr = np.atleast_1d(value)

    return arr


class SPECTRUM(FITSExtension):

    _HEADER_KEYWORDS = (
        ("EXTNAME", "SPECTRUM", "Extension name"),
        ("CONTENT", "OGIP PHA data", "File content"),
        ("HDUCLASS", "OGIP    ", "format conforms to OGIP standard"),
        ("HDUVERS", "1.1.0   ", "Version of format (OGIP memo CAL/GEN/92-002a)"),
        (
            "HDUDOC",
            "OGIP memos CAL/GEN/92-002 & 92-002a",
            "Documents describing the forma",
        ),
        ("HDUVERS1", "1.0.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUVERS2", "1.1.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUCLAS1", "SPECTRUM", "Extension contains spectral data  "),
        ("HDUCLAS2", "TOTAL ", ""),
        ("HDUCLAS3", "COUNTS ", ""),
        ("HDUCLAS4", "TYPE:II ", ""),
        ("FILTER", "", "Filter used"),
        ("CHANTYPE", "PI", "Channel type"),
        ("POISSERR", False, "Are the rates Poisson distributed"),
        ("DETCHANS", None, "Number of channels"),
        ("CORRSCAL", 1.0, ""),
        ("AREASCAL", 1.0, ""),
    )

    def __init__(
        self,
        tstart,
        telapse,
        channel,
        counts,
        quality,
        grouping,
        exposure,
        backscale,
        respfile,
        ancrfile,
        back_file=None,
        sys_err=None,
        stat_err=None,
        is_poisson=False,
    ):
        """
        Represents the SPECTRUM extension of a PHAII file.

        :param tstart: array of interval start times
        :param telapse: array of times elapsed since start
        :param channel: arrary of channel numbers
        :param counts: array of counts
        :param quality: array of OGIP quality values
        :param grouping: array of OGIP grouping values
        :param exposure: array of exposures
        :param backscale: array of backscale values
        :param respfile: array of associated response file names
        :param ancrfile: array of associate ancillary file names
        :param back_file: array of associated background file names
        :param sys_err: array of optional systematic errors
        :param stat_err: array of optional statistical errors (required of non poisson!)
        """

        n_spectra = len(tstart)

        data_list = [
            ("TSTART", tstart),
            ("TELAPSE", telapse),
            ("SPEC_NUM", np.arange(1, n_spectra + 1, dtype=np.int16)),
            ("CHANNEL", channel),
            ("COUNTS", counts),
            ("QUALITY", quality),
            ("BACKSCAL", backscale),
            ("GROUPING", grouping),
            ("EXPOSURE", exposure),
            ("RESPFILE", respfile),
            ("ANCRFILE", ancrfile),
        ]

        if back_file is not None:

            data_list.append(("BACKFILE", back_file))

        if stat_err is not None:

            if is_poisson:

                log.error(
                    "Tying to enter STAT_ERR error but have POISSERR set true")

                raise RuntimeError()
            data_list.append(("STAT_ERR", stat_err))

        if sys_err is not None:

            data_list.append(("SYS_ERR", sys_err))

        super(SPECTRUM, self).__init__(tuple(data_list), self._HEADER_KEYWORDS)

        self.hdu.header.set("POISSERR", is_poisson)

class PHAII(FITSFile):
    def __init__(
        self,
        filename: str,
        tstart: np.ndarray,
        telapse: np.ndarray,
        channel: np.ndarray,
        counts: np.ndarray,
        quality: np.ndarray,
        # grouping: np.ndarray, # Commenting for time being (693)
        exposure: np.ndarray,
        # backscale: np.ndarray, # Commenting for time being (695)
        respfile: np.ndarray,
        # ancrfile: np.ndarray, # Commenting for time being (697)
        back_file: Optional[np.ndarray] = None,
        sys_err: Optional[np.ndarray] = None,
        stat_err: Optional[np.ndarray] = None,
        is_poisson: bool = False,
    ):
        """

        A generic PHAII fits file

        :param instrument_name: name of the instrument
        :param telescope_name: name of the telescope
        :param tstart: array of interval start times
        :param telapse: array of times elapsed since start
        :param channel: arrary of channel numbers
        :param counts: array of counts
        :param quality: array of OGIP quality values
        :param grouping: array of OGIP grouping values
        :param exposure: array of exposures
        :param backscale: array of backscale values
        :param respfile: array of associated response file names
        :param ancrfile: array of associate ancillary file names
        :param back_file: array of associated background file names
        :param sys_err: array of optional systematic errors
        :param stat_err: array of optional statistical errors (required of non poisson!)
        """

        # collect the data so that we can have a general
        # extension builder

        self._filename = filename
        self._tstart = _atleast_1d_with_dtype(tstart, np.float32) * u.s
        self._telapse = _atleast_1d_with_dtype(telapse, np.float32) * u.s
        self._channel = _atleast_2d_with_dtype(channel, np.int16)
        self._counts = _atleast_2d_with_dtype(counts, np.float32) #* 1.0 / u.s
        self._exposure = _atleast_1d_with_dtype(exposure, np.float32) * u.s
        self._quality = _atleast_2d_with_dtype(quality, np.int16)
        # self._grouping = _atleast_2d_with_dtype(grouping, np.int16)
        # self._backscale = _atleast_1d_with_dtype(backscale, np.float32)
        self._respfile = _atleast_1d_with_dtype(respfile, str)
        # self._ancrfile = _atleast_1d_with_dtype(ancrfile, str)

        if sys_err is not None:

            self._sys_err = _atleast_2d_with_dtype(sys_err, np.float32)

        else:

            self._sys_err = sys_err

        if stat_err is not None:

            self._stat_err = _atleast_2d_with_dtype(stat_err, np.float32)

        else:

            self._stat_err = stat_err

        if back_file is not None:

            self._back_file = _atleast_1d_with_dtype(back_file, str)
        else:

            self._back_file = np.array(["NONE"] * self._tstart.shape[0])

        # Create the SPECTRUM extension

        spectrum_extension = SPECTRUM(
            self._tstart,
            self._telapse,
            self._channel,
            self._counts,
            self._quality,
            self._grouping,
            self._exposure,
            self._backscale,
            self._respfile,
            self._ancrfile,
            back_file=self._back_file,
            sys_err=self._sys_err,
            stat_err=self._stat_err,
            is_poisson=is_poisson,
        )

        # Set telescope and instrument name

        spectrum_extension.hdu.header.set("TELESCOP", 'AL1')
        spectrum_extension.hdu.header.set("INSTRUME", 'SoLEXS')
        spectrum_extension.hdu.header.set("DETCHANS", len(self._channel[0]))

        super(PHAII, self).__init__(fits_extensions=[spectrum_extension])

        self.primary_header_update()

    @classmethod
    def from_time_series(cls, time_series, use_poly=False):

        pha_information = time_series.get_information_dict(use_poly)

        is_poisson = True

        if use_poly:

            is_poisson = False

        return PHAII(
            instrument_name=pha_information["instrument"],
            telescope_name=pha_information["telescope"],
            tstart=pha_information["tstart"],
            telapse=pha_information["telapse"],
            channel=pha_information["channel"],
            counts=pha_information["counts"],
            stat_err=pha_information["counts error"],
            quality=pha_information["quality"].to_ogip(),
            grouping=pha_information["grouping"],
            exposure=pha_information["exposure"],
            backscale=1.0,
            respfile=None,  # pha_information['response_file'],
            ancrfile=None,
            is_poisson=is_poisson,
        )

    @classmethod
    def from_fits_file(cls, fits_file):

        with fits.open(fits_file) as f:

            if "SPECTRUM" in f:
                spectrum_extension = f["SPECTRUM"]
            else:
                log.warning("unable to find SPECTRUM extension: not OGIP PHA!")

                spectrum_extension = None

                for extension in f:
                    hduclass = extension.header.get("HDUCLASS")
                    hduclas1 = extension.header.get("HDUCLAS1")

                    if hduclass == "OGIP" and hduclas1 == "SPECTRUM":
                        spectrum_extension = extension
                        log.warning(
                            "File has no SPECTRUM extension, but found a spectrum in extension %s"
                            % (spectrum_extension.header.get("EXTNAME"))
                        )
                        spectrum_extension.header["EXTNAME"] = "SPECTRUM"
                        break

            spectrum = FITSExtension.from_fits_file_extension(
                spectrum_extension)

            out = FITSFile(primary_hdu=f["PRIMARY"],
                           fits_extensions=[spectrum])

        return out

    @property
    def instrument(self):
        return

    def primary_header_update(self):
        _PRIMARY_HEADER_KEYWORDS = (
            ("MISSION" , 'ADITYA L-1', 'Name of mission/satellite'),
            ("TELESCOP", 'AL1' , 'Name of mission/satellite'),
            ("INSTRUME", 'SoLEXS'      , 'Name of Instrument/detector'),
            ("ORIGIN"  , 'SoLEXSPOC'       , 'Source of FITS file'),
            ("CREATOR" , 'solexs_pipeline '  , 'Creator of file'),
            ("FILENAME", self._filename            , 'Name of file'),
            ("CONTENT" , 'Type II PHA file' , 'File content'),
            ("DATE", datetime.datetime.now().strftime("%Y-%m-%d"), 'Creation Date'),
        )

        primary_header = self._hdu_list[0].header

        for k in _PRIMARY_HEADER_KEYWORDS:
            primary_header.append(k)

        self._hdu_list[0].header = primary_header



class SPECTRUM_INTERM(FITSExtension):

    _HEADER_KEYWORDS = (
        ("EXTNAME", "SPECTRUM", "Extension name"),
        ("CONTENT", "OGIP PHA data", "File content"),
        ("HDUCLASS", "OGIP    ", "format conforms to OGIP standard"),
        ("HDUVERS", "1.1.0   ", "Version of format (OGIP memo CAL/GEN/92-002a)"),
        (
            "HDUDOC",
            "OGIP memos CAL/GEN/92-002 & 92-002a",
            "Documents describing the forma",
        ),
        ("HDUVERS1", "1.0.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUVERS2", "1.1.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUCLAS1", "SPECTRUM", "Extension contains spectral data  "),
        ("HDUCLAS2", "TOTAL ", ""),
        ("HDUCLAS3", "COUNTS ", ""),
        ("HDUCLAS4", "TYPE:II ", ""),
        ("FILTER", "", "Filter used"),
        ("CHANTYPE", "PHA", "Channel type"),
        ("POISSERR", False, "Are the rates Poisson distributed"),
        ("DETCHANS", None, "Number of channels"),
        ("CORRSCAL", 1.0, ""),
        ("AREASCAL", 1.0, ""),
    )

    def __init__(
        self,
        tstart,
        telapse,
        channel,
        counts,
        quality,
        exposure,
        e_min,
        e_max,
        back_file=None,
        sys_err=None,
        stat_err=None,
        is_poisson=False,
    ):
 

        n_spectra = len(tstart)

        data_list = [
            ("TSTART", tstart),
            ("TELAPSE", telapse),
            ("SPEC_NUM", np.arange(1, n_spectra + 1, dtype=np.int16)),
            ("CHANNEL", channel),
            ("COUNTS", counts),
            ("QUALITY", quality),
            ("EXPOSURE", exposure),
            ("E_MIN", e_min),
            ("E_MAX", e_max),
        ]

        if back_file is not None:

            data_list.append(("BACKFILE", back_file))

        if stat_err is not None:

            if is_poisson:

                log.error(
                    "Tying to enter STAT_ERR error but have POISSERR set true")

                raise RuntimeError()
            data_list.append(("STAT_ERR", stat_err))

        if sys_err is not None:

            data_list.append(("SYS_ERR", sys_err))

        super(SPECTRUM_INTERM, self).__init__(tuple(data_list), self._HEADER_KEYWORDS)

        self.hdu.header.set("POISSERR", is_poisson)

class PHAII_INTERM(FITSFile):
    def __init__(
        self,
        filename: str,
        tstart: np.ndarray,
        telapse: np.ndarray,
        channel: np.ndarray,
        counts: np.ndarray,
        quality: np.ndarray,
        exposure: np.ndarray,
        e_min: np.ndarray,
        e_max: np.ndarray,
        back_file: Optional[np.ndarray] = None,
        sys_err: Optional[np.ndarray] = None,
        stat_err: Optional[np.ndarray] = None,
        is_poisson: bool = False,
    ):
        """

        A generic PHAII fits file

        :param instrument_name: name of the instrument
        :param telescope_name: name of the telescope
        :param tstart: array of interval start times
        :param telapse: array of times elapsed since start
        :param channel: arrary of channel numbers
        :param counts: array of counts
        :param quality: array of OGIP quality values
        :param grouping: array of OGIP grouping values
        :param exposure: array of exposures
        :param backscale: array of backscale values
        :param respfile: array of associated response file names
        :param ancrfile: array of associate ancillary file names
        :param back_file: array of associated background file names
        :param sys_err: array of optional systematic errors
        :param stat_err: array of optional statistical errors (required of non poisson!)
        """

        # collect the data so that we can have a general
        # extension builder

        self._filename = filename
        self._tstart = _atleast_1d_with_dtype(tstart, np.float32) * u.s
        self._telapse = _atleast_1d_with_dtype(telapse, np.float32) * u.s
        self._channel = _atleast_2d_with_dtype(channel, np.int16)
        self._counts = _atleast_2d_with_dtype(counts, np.float32) #* 1.0 / u.s
        self._exposure = _atleast_1d_with_dtype(exposure, np.float32) * u.s
        self._quality = _atleast_2d_with_dtype(quality, np.int16)
        self._e_min = _atleast_2d_with_dtype(e_min, np.float32) * u.keV
        self._e_max = _atleast_2d_with_dtype(e_max, np.float32) * u.keV

        if sys_err is not None:

            self._sys_err = _atleast_2d_with_dtype(sys_err, np.float32)

        else:

            self._sys_err = sys_err

        if stat_err is not None:

            self._stat_err = _atleast_2d_with_dtype(stat_err, np.float32)

        else:

            self._stat_err = stat_err

        if back_file is not None:

            self._back_file = _atleast_1d_with_dtype(back_file, str)
        else:

            self._back_file = np.array(["NONE"] * self._tstart.shape[0])

        # Create the SPECTRUM extension

        spectrum_extension = SPECTRUM_INTERM(
            self._tstart,
            self._telapse,
            self._channel,
            self._counts,
            self._quality,
            self._exposure,
            self._e_min,
            self._e_max,
            back_file=self._back_file,
            sys_err=self._sys_err,
            stat_err=self._stat_err,
            is_poisson=is_poisson,
        )

        # Set telescope and instrument name

        spectrum_extension.hdu.header.set("TELESCOP", 'AL1')
        spectrum_extension.hdu.header.set("INSTRUME", 'SoLEXS')
        spectrum_extension.hdu.header.set("DETCHANS", len(self._channel[0]))

        super(PHAII_INTERM, self).__init__(fits_extensions=[spectrum_extension])

        self.primary_header_update()

    @property
    def instrument(self):
        return

    def primary_header_update(self):
        _PRIMARY_HEADER_KEYWORDS = (
            ("MISSION" , 'ADITYA L-1', 'Name of mission/satellite'),
            ("TELESCOP", 'AL1' , 'Name of mission/satellite'),
            ("INSTRUME", 'SoLEXS'      , 'Name of Instrument/detector'),
            ("ORIGIN"  , 'SoLEXSPOC'       , 'Source of FITS file'),
            ("CREATOR" , 'solexs_pipeline '  , 'Creator of file'),
            ("FILENAME",  self._filename            , 'Name of file'),
            ("CONTENT" , 'Type II PHA file' , 'File content'),
            ("DATE"    ,  datetime.datetime.now().strftime("%Y-%m-%d") , 'Creation Date'),
        )

        primary_header = self._hdu_list[0].header

        for k in _PRIMARY_HEADER_KEYWORDS:
            primary_header.append(k)

        self._hdu_list[0].header = primary_header


# From Rwitika        
class HK(FITSExtension):

    _HEADER_KEYWORDS = (
        ("EXTNAME", "HK", "Extension name"),
        ("HDUCLASS", "OGIP    ", "format conforms to OGIP standard"),
        ("HDUCLAS1", "HK    ", "Extension contains housekeeping parameters"),
        ("HDUCLAS2", "PKT    ", "Each row corresponds to info of a data 'packet'"),
        ("HDUVERS", "1.1.0   ", "Version of format (OGIP memo CAL/GEN/92-002a)"),
        (
            "HDUDOC",
            "OGIP memos 93-003 & 94-003",
            "Documents describing the format",
        ),
        ("HDUVERS1", "1.0.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUVERS2", "1.1.0   ", "Obsolete - included for backwards compatibility"),
    )

    def __init__(self, hk_arr, hk_colnames):
        """
        Represents the HK extension of a housekeeping FITS file
        
        """
        
        hkdata_list = []
        for name in hk_colnames:
            hkdata_list.append((name.upper(), hk_arr[name]))
        
        data_tuple = tuple(hkdata_list)

        super(HK, self).__init__(data_tuple, self._HEADER_KEYWORDS)

class HOUSEKEEPING(FITSFile):
    """
    HK parameters corresponding to each data packet of a particular raw (binary) file.

    """

    def __init__(
        self,
        filename,
        hk_arr,
        hk_colnames
    ):

        # Make sure that the provided iterables are of the right type for the FITS format
        ## ADD this!!!

        # Create EVENTS extension
        hk_ext = HK(hk_arr, hk_colnames)

        self._filename = filename

        # Create FITS file
        super(HOUSEKEEPING, self).__init__(fits_extensions=[hk_ext])

    def primary_header_update(self):
        _PRIMARY_HEADER_KEYWORDS = (
            ("MISSION" , 'ADITYA L-1', 'Name of mission/satellite'),
            ("TELESCOP", 'AL1' , 'Name of mission/satellite'),
            ("INSTRUME", 'SoLEXS'      , 'Name of Instrument/detector'),
            ("ORIGIN"  , 'SoLEXSPOC'       , 'Source of FITS file'),
            ("CREATOR" , 'solexs_pipeline '  , 'Creator of file'),
            ("FILENAME",  self._filename            , 'Name of file'),
            ("CONTENT" , 'Housekeeping Data file' , 'File content'),
            ("DATE"    ,  datetime.datetime.now().strftime("%Y-%m-%d") , 'Creation Date'),
        )

        primary_header = self._hdu_list[0].header

        for k in _PRIMARY_HEADER_KEYWORDS:
            primary_header.append(k)

        self._hdu_list[0].header = primary_header


class RATE_INTERM(FITSExtension):

    _HEADER_KEYWORDS = (
        ("EXTNAME", "RATE", "Extension name"),
        ("CONTENT", "LIGHT CURVE", "File content"),
        ("HDUCLASS", "OGIP    ", "format conforms to OGIP standard"),
        ("HDUVERS", "1.1.0   ", "Version of format (OGIP memo CAL/GEN/92-002a)"),
        (
            "HDUDOC",
            "OGIP memos CAL/GEN/92-007",
            "Documents describing the format",
        ),
        ("HDUVERS1", "1.0.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUVERS2", "1.1.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUCLAS1", "LIGHTCURVE", "Extension contains spectral data  "),
        ("HDUCLAS2", "TOTAL ", ""),
        ("HDUCLAS3", "COUNTS ", ""),
        ("FILTER", "", "Filter used"),
    )

    def __init__(
        self,
        tm,
        counts_low,
        counts_med,
        counts_high,
        counts_all
        # countrs_error=False,
    ):
 


        data_list = [
            ("TIME", tm),
            ("COUNTS_LOW", counts_low),
            ("COUNTS_MED", counts_med),
            ("COUNTS_HIGH", counts_high),
            ("COUNTS_ALL", counts_all),
            # ("E_MAX", e_max),
        ]


        super(RATE_INTERM, self).__init__(tuple(data_list), self._HEADER_KEYWORDS)


class ENEBAND_INTERM(FITSExtension):

    _HEADER_KEYWORDS = (
        ("EXTNAME", "ENEBAND", "Extension name"),
        # ("CONTENT", "LIGHT CURVE", "File content"),
        ("HDUCLASS", "OGIP    ", "format conforms to OGIP standard"),
        ("HDUVERS", "1.1.0   ", "Version of format (OGIP memo CAL/GEN/92-002a)"),
        (
            "HDUDOC",
            "OGIP memos CAL/GEN/92-007",
            "Documents describing the format",
        ),
        ("HDUVERS1", "1.0.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUVERS2", "1.1.0   ", "Obsolete - included for backwards compatibility"),
        # ("HDUCLAS1", "LIGHTCURVE", "Extension contains spectral data  "),
        # ("HDUCLAS2", "TOTAL ", ""),
        # ("HDUCLAS3", "COUNTS ", ""),
        # ("FILTER", "", "Filter used"),
    )

    def __init__(
        self,
        minchan,
        maxchan,
        # countrs_error=False,
    ):
 


        data_list = [
            ("MINCHAN", minchan),
            ("MAXCHAN", maxchan),
            # ("E_MAX", e_max),
        ]


        super(ENEBAND_INTERM, self).__init__(tuple(data_list), self._HEADER_KEYWORDS)



class LC_INTERM(FITSFile):
    def __init__(
        self,
        filename: str,
        tm: np.ndarray,
        counts_low: np.ndarray,
        counts_med: np.ndarray,
        counts_high: np.ndarray,
        counts_all: np.ndarray,
        minchan: np.ndarray,
        maxchan: np.ndarray,
        is_poisson: bool = False,
    ):
        """

        A generic PHAII fits file

        :param instrument_name: name of the instrument
        :param telescope_name: name of the telescope
        :param tstart: array of interval start times
        :param telapse: array of times elapsed since start
        :param channel: arrary of channel numbers
        :param counts: array of counts
        :param quality: array of OGIP quality values
        :param grouping: array of OGIP grouping values
        :param exposure: array of exposures
        :param backscale: array of backscale values
        :param respfile: array of associated response file names
        :param ancrfile: array of associate ancillary file names
        :param back_file: array of associated background file names
        :param sys_err: array of optional systematic errors
        :param stat_err: array of optional statistical errors (required of non poisson!)
        """

        # collect the data so that we can have a general
        # extension builder

        self._filename = filename
        self._time = _atleast_1d_with_dtype(tm, np.float32) * u.s
        # self._telapse = _atleast_1d_with_dtype(telapse, np.float32) * u.s
        # self._channel = _atleast_2d_with_dtype(channel, np.int16)
        self._counts_low = _atleast_1d_with_dtype(counts_low, np.int32) #* 1.0 / u.s
        self._counts_med = _atleast_1d_with_dtype(counts_med, np.int32) #* 1.0 / u.s
        self._counts_high = _atleast_1d_with_dtype(counts_high, np.int32) #* 1.0 / u.s
        self._counts_all = _atleast_1d_with_dtype(counts_all, np.int32) #* 1.0 / u.s
        # self._exposure = _atleast_1d_with_dtype(exposure, np.float32) * u.s
        # self._quality = _atleast_2d_with_dtype(quality, np.int16)
        # self._e_min = _atleast_2d_with_dtype(e_min, np.float32) * u.keV
        # self._e_max = _atleast_2d_with_dtype(e_max, np.float32) * u.keV
        self._minchan = _atleast_1d_with_dtype(minchan,np.int16)
        self._maxchan = _atleast_1d_with_dtype(maxchan,np.int16)


        # Create the RATE extension

        rate_extension = RATE_INTERM(
            self._time,
            self._counts_low,
            self._counts_med,
            self._counts_high,
            self._counts_all,
        )

        eneband_extension = ENEBAND_INTERM(self._minchan,self._maxchan)
        # Set telescope and instrument name

        rate_extension.hdu.header.set("TELESCOP", 'AL1')
        rate_extension.hdu.header.set("INSTRUME", 'SoLEXS')
        rate_extension.hdu.header.set("NUMBAND", '4')

        super(LC_INTERM, self).__init__(fits_extensions=[eneband_extension,rate_extension])

        self.primary_header_update()

    @property
    def instrument(self):
        return

    def primary_header_update(self):
        _PRIMARY_HEADER_KEYWORDS = (
            ("MISSION" , 'ADITYA L-1', 'Name of mission/satellite'),
            ("TELESCOP", 'AL1' , 'Name of mission/satellite'),
            ("INSTRUME", 'SoLEXS'      , 'Name of Instrument/detector'),
            ("ORIGIN"  , 'SoLEXSPOC'       , 'Source of FITS file'),
            ("CREATOR" , 'solexs_pipeline '  , 'Creator of file'),
            ("FILENAME",  self._filename            , 'Name of file'),
            ("CONTENT" , 'LIGHT CURVE' , 'File content'),
            ("DATE"    ,  datetime.datetime.now().strftime("%Y-%m-%d") , 'Creation Date'),
        )

        primary_header = self._hdu_list[0].header

        for k in _PRIMARY_HEADER_KEYWORDS:
            primary_header.append(k)

        self._hdu_list[0].header = primary_header

# https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/rates/ogip_93_003/ogip_93_003.html
class RATE(FITSExtension):

    _HEADER_KEYWORDS = (
        ("EXTNAME", "RATE", "Extension name"),
        ("CONTENT", "LIGHT CURVE", "File content"),
        ("HDUCLASS", "OGIP    ", "format conforms to OGIP standard"),
        ("HDUVERS", "1.1.0   ", "Version of format (OGIP memo CAL/GEN/92-002a)"),
        (
            "HDUDOC",
            "OGIP memos CAL/GEN/92-007",
            "Documents describing the format",
        ),
        ("HDUVERS1", "1.0.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUVERS2", "1.1.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUCLAS1", "LIGHTCURVE", "Extension contains spectral data  "),
        ("HDUCLAS2", "TOTAL ", ""),
        ("HDUCLAS3", "COUNTS ", ""),
        ("FILTER", "", "Filter used"),
    )

    def __init__(
        self,
        tm,
        counts_low,
        counts_med,
        counts_high,
        counts_all
        # countrs_error=False,
    ):

        data_list = [
            ("TIME", tm),
            ("COUNTS_LOW", counts_low),
            ("COUNTS_MED", counts_med),
            ("COUNTS_HIGH", counts_high),
            ("COUNTS_ALL", counts_all),
            # ("E_MAX", e_max),
        ]

        super(RATE, self).__init__(
            tuple(data_list), self._HEADER_KEYWORDS)


class ENEBAND(FITSExtension):

    _HEADER_KEYWORDS = (
        ("EXTNAME", "ENEBAND", "Extension name"),
        # ("CONTENT", "LIGHT CURVE", "File content"),
        ("HDUCLASS", "OGIP    ", "format conforms to OGIP standard"),
        ("HDUVERS", "1.1.0   ", "Version of format (OGIP memo CAL/GEN/92-002a)"),
        (
            "HDUDOC",
            "OGIP memos CAL/GEN/92-007",
            "Documents describing the format",
        ),
        ("HDUVERS1", "1.0.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUVERS2", "1.1.0   ", "Obsolete - included for backwards compatibility"),
        # ("HDUCLAS1", "LIGHTCURVE", "Extension contains spectral data  "),
        # ("HDUCLAS2", "TOTAL ", ""),
        # ("HDUCLAS3", "COUNTS ", ""),
        # ("FILTER", "", "Filter used"),
    )

    def __init__(
        self,
        minchan: np.ndarray,
        maxchan: np.ndarray,
        # countrs_error=False,
    ):
        
        self._minchan = _atleast_2d_with_dtype(minchan,np.int16)
        self._maxchan = _atleast_2d_with_dtype(maxchan, np.int16)

        data_list = [
            ("MINCHAN", self._minchan),
            ("MAXCHAN", self._maxchan),
            # ("E_MAX", e_max),
        ]

        super(ENEBAND, self).__init__(
            tuple(data_list), self._HEADER_KEYWORDS)


class LC(FITSFile):
    def __init__(
        self,
        filename: str,
        tm: np.ndarray,
        counts_low: np.ndarray,
        counts_med: np.ndarray,
        counts_high: np.ndarray,
        counts_all: np.ndarray,
        minchan: np.ndarray,
        maxchan: np.ndarray,
        is_poisson: bool = False,
    ):
        """

        A generic PHAII fits file

        :param instrument_name: name of the instrument
        :param telescope_name: name of the telescope
        :param tstart: array of interval start times
        :param telapse: array of times elapsed since start
        :param channel: arrary of channel numbers
        :param counts: array of counts
        :param quality: array of OGIP quality values
        :param grouping: array of OGIP grouping values
        :param exposure: array of exposures
        :param backscale: array of backscale values
        :param respfile: array of associated response file names
        :param ancrfile: array of associate ancillary file names
        :param back_file: array of associated background file names
        :param sys_err: array of optional systematic errors
        :param stat_err: array of optional statistical errors (required of non poisson!)
        """

        # collect the data so that we can have a general
        # extension builder

        self._filename = filename
        self._time = _atleast_1d_with_dtype(tm, np.float32) * u.s
        # self._telapse = _atleast_1d_with_dtype(telapse, np.float32) * u.s
        # self._channel = _atleast_2d_with_dtype(channel, np.int16)
        self._counts_low = _atleast_1d_with_dtype(
            counts_low, np.int32)  # * 1.0 / u.s
        self._counts_med = _atleast_1d_with_dtype(
            counts_med, np.int32)  # * 1.0 / u.s
        self._counts_high = _atleast_1d_with_dtype(
            counts_high, np.int32)  # * 1.0 / u.s
        self._counts_all = _atleast_1d_with_dtype(
            counts_all, np.int32)  # * 1.0 / u.s
        # self._exposure = _atleast_1d_with_dtype(exposure, np.float32) * u.s
        # self._quality = _atleast_2d_with_dtype(quality, np.int16)
        # self._e_min = _atleast_2d_with_dtype(e_min, np.float32) * u.keV
        # self._e_max = _atleast_2d_with_dtype(e_max, np.float32) * u.keV
        self._minchan = _atleast_1d_with_dtype(minchan, np.int16)
        self._maxchan = _atleast_1d_with_dtype(maxchan, np.int16)

        # Create the RATE extension

        rate_extension = RATE(
            self._time,
            self._counts_low,
            self._counts_med,
            self._counts_high,
            self._counts_all,
        )

        eneband_extension = ENEBAND(self._minchan, self._maxchan)
        # Set telescope and instrument name

        rate_extension.hdu.header.set("TELESCOP", 'AL1')
        rate_extension.hdu.header.set("INSTRUME", 'SoLEXS')
        rate_extension.hdu.header.set("NUMBAND", '4')

        super(LC, self).__init__(
            fits_extensions=[eneband_extension, rate_extension])

        self.primary_header_update()

    @property
    def instrument(self):
        return

    def primary_header_update(self):
        _PRIMARY_HEADER_KEYWORDS = (
            ("MISSION", 'ADITYA L-1', 'Name of mission/satellite'),
            ("TELESCOP", 'AL1', 'Name of mission/satellite'),
            ("INSTRUME", 'SoLEXS', 'Name of Instrument/detector'),
            ("ORIGIN", 'SoLEXSPOC', 'Source of FITS file'),
            ("CREATOR", 'solexs_pipeline ', 'Creator of file'),
            ("FILENAME",  self._filename, 'Name of file'),
            ("CONTENT", 'LIGHT CURVE', 'File content'),
            ("DATE",  datetime.datetime.now().strftime("%Y-%m-%d"), 'Creation Date'),
        )

        primary_header = self._hdu_list[0].header

        for k in _PRIMARY_HEADER_KEYWORDS:
            primary_header.append(k)

        self._hdu_list[0].header = primary_header
