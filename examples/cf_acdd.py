"""Functions to simplify CF and ACDD compliance."""

from __future__ import division, print_function
import datetime
import sys
import os
try:
    from shlex import quote
except ImportError:
    from pipes import quote
from pwd import getpwuid
import subprocess
import socket

import dateutil.tz

UTC = dateutil.tz.tzutc()
UDUNITS_DATE = "%Y-%m-%d %H:%M:%S%z"
ACDD_DATE = "%Y-%m-%dT%H:%M:%S%z"
CALENDAR = "standard"
RUN_DATE = datetime.datetime.now(tz=UTC)
HOST = socket.gethostbyaddr(socket.gethostbyname(os.uname()[1]))[0]
MAIN_HOST = ".".join(HOST.split(".")[-3:])
COMMAND_LINE = " ".join(quote(arg) for arg in sys.argv)


def global_attributes_dict():
    """Set global attributes required by conventions.

    Currently CF-1.6 and ACDD-1.3.

    Returns
    -------
    global_atts: dict
        Still needs title, summary, source, creator_institution,
        product_version, references, cdm_data_type, institution,
        geospatial_vertical_{min,max,positive,units}, ...

    References
    ----------
    CF Conventions document: cfconventions.org
    ACDD document: http://wiki.esipfed.org/index.php/Category:Attribute_Conventions_Dataset_Discovery
    NCEI Templates: https://www.nodc.noaa.gov/data/formats/netcdf/v2.0/
    """
    username = getpwuid(os.getuid())[0]
    global_atts = dict(
        Conventions="CF-1.6 ACDD-1.3",
        standard_name_vocabulary="CF Standard Name Table v32",
        history=("{now:{date_fmt:s}}: Created by {progname:s} "
                 "with command line: {cmd_line:s}").format(
            now=RUN_DATE, date_fmt=UDUNITS_DATE, progname=sys.argv[0],
            cmd_line=COMMAND_LINE,
        ),
        source=("Created by {progname:s} "
                "with command line: {cmd_line:s}").format(
            progname=sys.argv[0], cmd_line=COMMAND_LINE,
        ),
        date_created="{now:{date_fmt:s}}".format(
            now=RUN_DATE, date_fmt=ACDD_DATE),
        date_modified="{now:{date_fmt:s}}".format(
            now=RUN_DATE, date_fmt=ACDD_DATE),
        date_metadata_modified="{now:{date_fmt:s}}".format(
            now=RUN_DATE, date_fmt=ACDD_DATE),
        creator_name=username,
        creator_email="{username:s}@{host:s}".format(
            username=username,
            host=MAIN_HOST,
        ),
        creator_institution=MAIN_HOST,
    )

    try:
        global_atts["conda_packages"] = subprocess.check_output(
            # Full urls including package, version, build, and MD5
            ["conda", "list", "--explicit", "--md5"],
            universal_newlines=True,
        )
    except OSError:
        pass

    try:
        global_atts["pip_packages"] = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            universal_newlines=True,
        )
    except OSError:
        pass

    return global_atts
