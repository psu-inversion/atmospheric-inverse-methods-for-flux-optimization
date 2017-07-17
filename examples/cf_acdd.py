"""Functions to simplify CF and ACDD compliance."""

from __future__ import division, print_function
import datetime
import sys
import os
from shlex import quote
from pwd import getpwuid

import dateutil.tz

UTC = dateutil.tz.tzutc()
UDUNITS_DATE = "%Y-%m-%d %H:%M:%S%z"
ACDD_DATE = "%Y-%m-%dT%H:%M:%S%z"
CALENDAR = "standard"
RUN_DATE = datetime.datetime.now(tz=UTC)

def global_attributes_dict():
    """Set global attributes required by conventions.

    Currently CF-1.6 and ACDD-1.3.

    Returns
    -------
    global_atts: dict
        Still needs title, summary, source, creator_institution, 
        product_version, references, cdm_data_type, institution, 
        geospatial_vertical_{min,max,positive,units}, ...
    """
    return dict(
        Conventions="CF-1.6 ACDD-1.3",
        standard_name_vocabulary="CF Standard Name Table v32",
        history="{now:{date_fmt:s}}: Created by {progname:s} with command line: {cmd_line:s}".format(
            now=RUN_DATE, date_fmt=UDUNITS_DATE, progname=sys.argv[0],
            cmd_line=" ".join(quote(arg) for arg in sys.argv)),
        date_created="{now:{date_fmt:s}}".format(
            now=RUN_DATE, date_fmt=ACDD_DATE),
        date_modified="{now:{date_fmt:s}}".format(
            now=RUN_DATE, date_fmt=ACDD_DATE),
        date_metadata_modified="{now:{date_fmt:s}}".format(
            now=RUN_DATE, date_fmt=ACDD_DATE),
        creator_name=getpwuid(os.getuid())[0],
        )
        
        
