#!/usr/bin/env python3
############################################################################
#
# MODULE:       t.rast.import.netcdf
# AUTHOR(S):    Stefan Blumentrath
# PURPOSE:      Import netCDF files that adhere to the CF convention as a
#               Space Time Raster Dataset (STRDS)
# COPYRIGHT:    (C) 2021 by stefan.blumentrath, and the GRASS Development Team
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
############################################################################

#%module
#% description: Import netCDF files that adhere to the CF convention as STRDS.
#% keyword: temporal
#% keyword: import
#% keyword: raster
#% keyword: time
#% keyword: netcdf
#%end

#%flag
#% key: a
#% description: Append to STRDS
#% guisection: Settings
#%end

#%flag
#% key: r
#% description: Import only within current region
#% guisection: Filter
#%end

#%flag
#% key: l
#% description: Link the raster files using r.external
#% guisection: Settings
#%end

#%flag
#% key: e
#% description: Extend location extents based on new dataset
#% guisection: Settings
#%end

#%flag
#% key: o
#% label: Override projection check (use current location's projection)
#% description: Assume that the dataset has same projection as the current location
#% guisection: Settings
#%end

#%option G_OPT_F_INPUT
#% key: input
#% type: string
#% required: yes
#% multiple: no
#% key_desc: Input file(s) ("-" = stdin)
#% description: URL or name of input netcdf-file ("-" = stdin)
#%end

#%option G_OPT_F_INPUT
#% key: bandref
#% type: string
#% required: no
#% multiple: no
#% key_desc: Input file with bandreference configuration ("-" = stdin)
#% description: File with mapping of variables or subdatasets to band references
#% guisection: Settings
#%end

#%option G_OPT_STRDS_OUTPUT
#% required: no
#% multiple: no
#% description: Name of the output space time raster dataset
#%end

#%option
#% key: end_time
#% label: Latest timestamp of temporal extent to include in the output
#% description: Timestamp of format "YYYY-MM-DD HH:MM:SS"
#% type: string
#% required: no
#% multiple: no
#% guisection: Filter
#%end

#%option
#% key: start_time
#% label: Earliest timestamp of temporal extent to include in the output
#% description: Timestamp of format "YYYY-MM-DD HH:MM:SS"
#% type: string
#% required: no
#% multiple: no
#% guisection: Filter
#%end

#%option
#% key: temporal_relations
#% label: Allowed temporal relation for temporal filtering
#% description: Allowed temporal relation between time dimension in the netCDF file and temporal window defined by start_time and end_time
#% type: string
#% required: no
#% multiple: yes
#% options: equal,during,contains,overlaps,overlapped,starts,started,finishes,finished
#% answer: equal,during,contains,overlaps,overlapped,starts,started,finishes,finished
#% guisection: Filter
#%end

#%option
#% key: resample
#% type: string
#% required: no
#% multiple: no
#% label: Resampling method when data is reprojected
#% options: nearest,bilinear,bilinear_f,bicubic,bicubic_f,cubicspline,lanczos,lanczos_f,min,Q1,average,med,Q3,max,mode
#% guisection: Settings
#%end

#%option
#% key: print
#% type: string
#% required: no
#% multiple: no
#% label: Print metadata and exit
#% options: extended, grass
#% guisection: Print
#%end

#%option G_OPT_M_COLR
#% description: Color table to assign to imported datasets
#% answer: viridis
#% guisection: Settings
#%end

#%option
#% key: memory
#% type: integer
#% required: no
#% multiple: no
#% key_desc: memory in MB
#% label: Maximum memory to be used (in MB)
#% description: Cache size for raster rows
#% answer: 300
#% guisection: Settings
#%end

#%option
#% key: nprocs
#% type: integer
#% required: no
#% multiple: no
#% key_desc: Number of cores
#% label: Number of cores to use during import
#% answer: 1
#% guisection: Settings
#%end

#%option G_OPT_F_SEP
#% guisection: Settings
#%end

#%rules
#% excludes: -o,resample
#% excludes: -l,-r
#% excludes: print,output
#% required: print,output
#%end

#%option
#% key: nodata
#% type: string
#% required: no
#% multiple: yes
#% key_desc: Source nodata
#% description: Comma separated list of values representing nodata in the input dataset
#%end

# Todo:
# Allow filtering based on metadata
# Support more VRT options (resolution, extent)
# Allow to print subdataset information as bandref json (useful defining custom bandreferences)
# - Make use of more metadata (units, scaling)

from copy import deepcopy
from datetime import datetime
from io import StringIO
from itertools import chain
from multiprocessing import Pool
import os
from pathlib import Path
import re
from subprocess import PIPE
import sys


import numpy as np

# import dateutil.parser as parser

from osgeo import gdal
import cf_units

import grass.script as gscript
import grass.temporal as tgis
from grass.pygrass.modules import Module, MultiModule, ParallelModuleQueue

# from grass.temporal. import update_from_registered_maps
from grass.temporal.register import register_maps_in_space_time_dataset
from grass.temporal.temporal_extent import TemporalExtent
from grass.temporal.datetime_math import datetime_to_grass_datetime_string

# Datasets may or may not contain subdatasets
# Datasets may contain several layers
# r.external registers all bands by default

resample_dict = {
    "gdal": {
        "nearest": "near",
        "bilinear": "bilinear",
        "bicubic": "cubic",
        "cubicspline": "cubicspline",
        "lanczos": "lanczos",
        "average": "average",
        "mode": "mode",
        "max": "max",
        "min": "min",
        "med": "med",
        "Q1": "Q1",
        "Q3": "Q3",
    },
    "grass": {
        "nearest": "nearest",
        "bilinear": "bilinear",
        "bicubic": "bicubic",
        "lanczos": "lanczos",
        "bilinear_f": "average",
        "bicubic_f": "bicubic_f",
        "lanczos_f": "lanczos_f",
    },
}

grass_version = list(map(int, gscript.version()["version"].split(".")[0:2]))


def legalize_name_string(string):
    """Replace conflicting characters with _"""
    legal_string = re.sub(r"[^\w\d-]+|[^\x00-\x7F]+|[ -/\\]+", "_", string)
    return legal_string


def get_time_dimensions(meta):
    """Extracts netcdf-cf compliant time dimensions from metadata using UDUNITS2"""
    if "NETCDF_DIM_time_VALUES" not in meta:
        return None
    time_values = np.fromstring(
        meta["NETCDF_DIM_time_VALUES"].strip("{").strip("}"), sep=",", dtype=np.float
    )
    time_dates = cf_units.num2date(
        time_values, meta["time#units"], meta["time#calendar"]
    )
    return time_dates


def parse_badref_conf(conf_file):
    """Read user provided mapping of subdatasets / variables to band references
    Return a dict with mapping, bands that are not mapped in this file are skipped
    from import"""
    if conf_file is None or conf_file == "":
        return None

    if grass_version[0] < 8:
        gscript.warning(
            _(
                "The band reference concept requires GRASS GIS version 8.0 or later.\n"
                "Ignoring the band reference configuration file <{conf_file}>".format(
                    conf_file=conf_file
                )
            )
        )
        return None

    bandref = {}
    if not os.access(options["bandref"], os.R_OK):
        gscript.fatal(
            _(
                "Cannot read configuration file <{conf_file}>".format(
                    conf_file=conf_file
                )
            )
        )
    # Lazy import GRASS GIS 8 function if needed
    from grass.lib.raster import Rast_legal_bandref

    with open(conf_file, "r") as c_file:
        configuration = c_file.read()
        for idx, line in enumerate(configuration.split("\n")):
            if line.startswith("#") or line == "":
                continue
            if len(line.split("=")) == 2:
                line = line.split("=")
                # Check if assigned band reference has legal a name
                if Rast_legal_bandref(line[1]) == 1:
                    bandref[line[0]] = line[1]
                else:
                    gscript.fatal(
                        _(
                            "Line {line_nr} in configuration file <{conf_file}> "
                            "contains an illegal band name".format(
                                line_nr=idx + 1, conf_file=conf_file
                            )
                        )
                    )
            else:
                gscript.fatal(
                    _(
                        "Invalid format of band reference configuration in file <{}>".format(
                            conf_file
                        )
                    )
                )

    return bandref


def get_metadata(netcdf_metadata, subdataset="", bandref=None):
    """Transform NetCDF metadata to GRASS metadata"""
    # title , history , institution , source , comment and references
    # netcdf_metadata = netcdf.GetMetadata()

    meta = {}
    # title is required metadata for netCDF-CF
    title = (
        netcdf_metadata["NC_GLOBAL#title"]
        if "NC_GLOBAL#title" in netcdf_metadata
        else ""
    )
    title += (
        ", {subdataset}: {long_name}, {method}".format(
            subdataset=subdataset,
            long_name=netcdf_metadata.get("{}#long_name".format(subdataset)),
            method=netcdf_metadata.get("{}#cell_methods".format(subdataset)),
        )
        if subdataset != ""
        else ""
    )
    title += (
        ", version: {}".format(netcdf_metadata["NC_GLOBAL#version"])
        if "NC_GLOBAL#version" in netcdf_metadata
        else ""
    )
    title += (
        ", type: {}".format(netcdf_metadata["NC_GLOBAL#type"])
        if "NC_GLOBAL#type" in netcdf_metadata
        else ""
    )
    meta["title"] = title
    # history is required metadata for netCDF-CF
    meta["history"] = netcdf_metadata.get(
        "NC_GLOBAL#history"
    )  # phrase Text to append to the next line of the map's metadata file
    meta["units"] = netcdf_metadata.get(
        "{}#units".format(subdataset)
    )  # string Text to use for map data units

    meta["vdatum"] = None  # string Text to use for map vertical datum
    meta["source1"] = netcdf_metadata.get("NC_GLOBAL#source")
    meta["source2"] = netcdf_metadata.get("NC_GLOBAL#institution")

    meta["description"] = "\n".join(
        map(
            str,
            filter(
                None,
                [
                    netcdf_metadata.get("NC_GLOBAL#summary"),
                    netcdf_metadata.get("NC_GLOBAL#comment"),
                    netcdf_metadata.get("NC_GLOBAL#references"),
                ],
            ),
        )
    )
    if bandref is not None:
        meta["bandref"] = bandref[subdataset]

    return meta


def check_projection_match(url):
    """Check if projections match"""
    try:
        projection_match = Module(
            "r.in.gdal", input=url, flags="j", stderr_=PIPE, stdout_=PIPE
        )
        projection_match = True
    except Exception:
        projection_match = False
    return projection_match


def get_import_type(url, projection_match, resample, flags_dict):
    """Define import type ("r.in.gdal", "r.import", "r.external")"""

    if not projection_match and not flags_dict["o"]:
        if not resample:
            gscript.warning(
                _(
                    "Projection for {} does not match the projection of the "
                    "current location, but no resampling method has been specified. "
                    "Using nearest neighbor method for resampling.".format(url)
                )
            )
            resample = "nearest"
        if flags_dict["l"]:
            gscript.warning(
                _(
                    "Cannot link {} directly, using a warped virtual raster through GDAL".format(
                        url
                    )
                )
            )
            import_type = "r.external"
            if resample not in resample_dict["gdal"]:
                gscript.fatal(
                    _(
                        "For re-projection with gdalwarp only the following "
                        "resample methods are allowed: {}".format(
                            ", ".join(list(resample_dict["gdal"].keys()))
                        )
                    )
                )
            resample = resample_dict["gdal"][resample]

        else:
            import_type = "r.import"
            if resample not in resample_dict["grass"]:
                gscript.fatal(
                    _(
                        "For re-projection with r.import only the following "
                        "resample methods are allowed: {}".format(
                            ", ".join(list(resample_dict["grass"].keys()))
                        )
                    )
                )
            resample = resample_dict["grass"][resample]
    elif flags_dict["l"]:
        import_type, resample = "r.external", None
    else:
        import_type, resample = "r.in.gdal", None

    return import_type, resample


def setup_temporal_filter(options_dict):
    """Gernerate temporal filter from input"""
    time_formats = {
        10: "%Y-%m-%d",
        19: "%Y-%m-%d %H:%M:%S",
    }
    kwargs = {}
    relations = options_dict["temporal_relations"].split(",")
    for time_ref in ["start_time", "end_time"]:
        if options_dict[time_ref]:
            if len(options_dict[time_ref]) not in time_formats:
                gscript.fatal(_("Unsupported datetime format in {}.".format(time_ref)))
            try:
                kwargs[time_ref] = datetime.strptime(
                    options_dict[time_ref], time_formats[len(options_dict[time_ref])]
                )
            except ValueError:
                gscript.fatal(_("Can not parse input in {}.".format(time_ref)))
        else:
            kwargs[time_ref] = None
    return TemporalExtent(**kwargs), relations


def apply_temporal_filter(ref_window, relations, start, end):
    """Apply temporal filter to time dimension"""
    if ref_window.start_time is None and ref_window.end_time is None:
        return True
    return (
        ref_window.temporal_relation(TemporalExtent(start_time=start, end_time=end))
        in relations
    )


def get_end_time(start_time_dimensions):
    """Compute end time from start time"""
    end_time_dimensions = None
    if len(start_time_dimensions) > 1:
        time_deltas = np.diff(start_time_dimensions)
        time_deltas = np.append(time_deltas, np.mean(time_deltas))
        end_time_dimensions = start_time_dimensions + time_deltas
    else:
        end_time_dimensions = start_time_dimensions
    return end_time_dimensions


# import or link data
def read_data(
    input_url,
    rastercount,
    metadata,
    options_dict,
    import_type,
    flags_dict,
    start_time_dimensions,
    end_time_dimensions,
    requested_time_dimensions,
    gisenv,
    projection_match,
    resamp,
    nodata,
    strds_name,
):
    """Import or link data and metadata"""
    maps = []
    queue = []

    imp_flags = "o" if flags_dict["o"] else ""
    # r.external [-feahvtr]
    # r.import [-enl]
    # r.in.gdal [-eflakcrp]
    is_subdataset = input_url.startswith("NETCDF")
    # Setup import module
    import_mod = Module(
        import_type,
        quiet=True,
        input=input_url
        if not is_subdataset
        else create_vrt(input_url, gisenv, resamp, nodata, projection_match),
        run_=False,
        finish_=False,
        flags=imp_flags,
        overwrite=gscript.overwrite(),
    )

    # import_mod.flags.l = ignore_crs
    if import_type != "r.external":
        import_mod.flags.a = True
        import_mod.flags.r = True
        import_mod.memory = options_dict["memory"]
    if import_type == "r.import":
        import_mod.resample = resamp
        # import_mod.resolution = options["resolution"]
        # import_mod.resolution_value = options["resolution_value"]
    if import_type == "r.in.gdal":
        import_mod.flags.r = flags_dict["r"]
        # import_mod.offset = options["offset"]
        # import_mod.num_digits = options["num_digits"]
    # Setup metadata module
    meta_mod = Module("r.support", quiet=True, run_=False, finish_=False, **metadata)
    # Setup timestamp module
    time_mod = Module("r.timestamp", quiet=True, run_=False, finish_=False)
    # Setup timestamp module
    color_mod = Module(
        "r.colors", quiet=True, color=options_dict["color"], run_=False, finish_=False
    )
    # Parallel module
    mapname_list = []
    infile = Path(input_url).name.split(":")
    mapname_list.append(legalize_name_string(infile[0]))
    if is_subdataset:
        mapname_list.append(legalize_name_string(infile[1]))

    for i, band in enumerate(requested_time_dimensions):
        mapname = "_".join(
            mapname_list + [start_time_dimensions[i].strftime("%Y_%m_%d")]
        )
        maps.append(
            "{map}@{mapset}|{start_time}|{end_time}|{bandref}".format(
                map=mapname,
                mapset=gisenv["MAPSET"],
                start_time=start_time_dimensions[i].strftime("%Y-%m-%d %H:%M:%S"),
                end_time=end_time_dimensions[i].strftime("%Y-%m-%d %H:%M:%S"),
                bandref="" if "bandref" not in metadata else metadata["bandref"],
            )
        )
        new_import = deepcopy(import_mod)
        new_import(band=band + 1, output=mapname)
        new_meta = deepcopy(meta_mod)
        new_meta(map=mapname)
        new_time = deepcopy(time_mod)
        new_time(
            map=mapname,
            date=datetime_to_grass_datetime_string(start_time_dimensions[i]),
        )
        new_color = deepcopy(color_mod)
        new_color(map=mapname)

        queue.append([new_import, new_meta, new_time, new_color])

    return strds_name, maps, queue


def create_vrt(subdataset_url, gisenv, resample, nodata, equal_proj):
    """Create a GDAL VRT for import"""
    vrt_dir = Path(gisenv["GISDBASE"]).joinpath(
        gisenv["LOCATION_NAME"], gisenv["MAPSET"], "gdal"
    )
    # current_region = gscript.region()
    if not vrt_dir.is_dir():
        vrt_dir.mkdir()
    vrt_name = str(
        vrt_dir.joinpath(
            "netcdf_{}.vrt".format(legalize_name_string(Path(subdataset_url).name))
        )
    )
    kwargs = {"format": "VRT"}
    if equal_proj:
        if nodata is not None:
            kwargs["noData"] = nodata
        vrt = gdal.Translate(
            vrt_name,
            subdataset_url,
            options=gdal.TranslateOptions(
                **kwargs
                # format="VRT",
                # stats=True,
                # outputType=gdal.GDT_Int16,
                # outputBounds=,
                # xRes=resolution,
                # yRes=resolution,
                # noData=nodata,
            ),
        )
    else:
        kwargs["dstSRS"] = gisenv["LOCATION_PROJECTION"]
        kwargs["resampleAlg"] = resample

        if nodata is not None:
            kwargs["srcNodata"] = nodata
        vrt = gdal.Warp(
            vrt_name,
            subdataset_url,
            options=gdal.WarpOptions(
                **kwargs
                # format="VRT",
                # outputType=gdal.GDT_Int16,
                # outputBounds
                # xRes=resolution,
                # yRes=resolution,
                # dstSRS=gisenv["LOCATION_PROJECTION"],
                # srcNodata=nodata,
                # resampleAlg=resample,
            ),
        )
    vrt = None
    vrt = vrt_name

    return vrt


def main():
    """run the main workflow"""

    input = options["input"].split(",")
    sep = gscript.utils.separator(options["separator"])

    valid_window, valid_relations = setup_temporal_filter(options)

    if options["nodata"]:
        try:
            nodata = " ".join(map(str, map(int, options["nodata"].split(","))))
        except Exception:
            gscript.fatal(_("Invalid input for nodata"))
    else:
        nodata = None

    if len(input) == 1:
        if input[0] == "-":
            input = sys.stdin.read().strip().split()
        elif not input[0].endswith(".nc"):
            try:
                with open(input[0], "r") as in_file:
                    input = in_file.read().strip().split()
            except IOError:
                gscript.fatal(_("Unable to read text from <{}>.".format(input[0])))

    input = [
        "/vsicurl/" + in_url if in_url.startswith("http") else in_url
        for in_url in input
    ]

    for in_url in input:
        if not in_url.endswith(".nc"):
            gscript.fatal(_("<{}> does not seem to be a NetCDF file".format(in_url)))

    bandref = parse_badref_conf(options["bandref"])

    # Get GRASS GIS environment info
    grass_env = dict(gscript.gisenv())

    # Get projection of the current location
    grass_env["LOCATION_PROJECTION"] = gscript.read_command(
        "g.proj", flags="wf"
    ).strip()

    # Initialize TGIS
    tgis.init()

    # Get existing STRDS
    dataset_list = tgis.list_stds.get_dataset_list(
        type="strds", temporal_type="absolute", columns="name"
    )
    existing_strds = (
        [row["name"] for row in dataset_list[grass_env["MAPSET"]]]
        if grass_env["MAPSET"] in dataset_list
        else []
    )

    # Check inputs
    # URL or file readable
    # STRDS exists / appcreation_dateend to
    # get basename from existing STRDS

    modified_strds = {}
    queued_modules = []
    queueing_input = []
    inputs = {}

    for in_url in input:
        # Check if file exists and readable
        try:
            ncdf = gdal.Open(in_url)
        except FileNotFoundError:
            gscript.fatal(_("Could not open <{}>".format(in_url)))

        # Get global metadata
        ncdf_metadata = ncdf.GetMetadata()

        # Get CF version
        cf_version = ncdf_metadata.get("NC_GLOBAL#Conventions")

        if cf_version is None or not cf_version.upper().startswith("CF"):
            gscript.warning(
                _(
                    "Input netCDF file does not adhere to CF-standard. Import may fail or be incorrect."
                )
            )

        # try:
        #     creation_date = parser.parse(ncdf_metadata.get("NC_GLOBAL#creation_date"))
        # except Exception:
        #     creation_date = None

        # Sub datasets containing variables have 3 dimensions (x,y,z)
        sds = [
            # SDS_ID, SDS_url, SDS_dimension
            [sds[0].split(":")[-1], sds[0], len(sds[1].split(" ")[0].split("x"))]
            for sds in ncdf.GetSubDatasets()
            if len(sds[1].split(" ")[0].split("x")) == 3
        ]

        # Filter based on bandref if provided
        if bandref is not None:
            sds = [s for s in sds if s[0] in bandref.keys()]

        # Open subdatasets to get metadata
        if len(sds) > 0:  # and ncdf.RasterCount == 0:
            sds = [[gdal.Open(s[1])] + s for s in sds]
        elif len(sds) == 0 and ncdf.RasterCount == 0:
            gscript.warning(_("No data to import from file {}").format(in_url))
        else:
            # Check raster layers
            sds = [ncdf, "", "", 0]

        # Extract metadata

        # Collect relevant inputs in a dictionary
        inputs[in_url] = {}
        inputs[in_url]["sds"] = [
            {
                "id": sd[1],
                "url": sd[0].GetDescription(),
                "grass_metadata": get_metadata(sd[0].GetMetadata(), sd[1], bandref),
                "extended_metadata": sd[0].GetMetadata(),
                "time_dimensions": get_time_dimensions(sd[0].GetMetadata()),
                "rastercount": sd[0].RasterCount,
            }
            for sd in sds
            if "NETCDF_DIM_time_VALUES" in sd[0].GetMetadata()
        ]

        # Close open GDAL datasets
        sds = None

        # Apply temporal filter
        for sd in inputs[in_url]["sds"]:
            end_times = get_end_time(sd["time_dimensions"])
            requested_time_dimensions = np.array(
                [
                    apply_temporal_filter(
                        valid_window, valid_relations, start, end_times[idx]
                    )
                    for idx, start in enumerate(sd["time_dimensions"])
                ]
            )
            if not requested_time_dimensions.any():
                gscript.warning(
                    _(
                        "Nothing to import from subdataset {s} in {f}".format(
                            s=sd["id"], f=sd["url"]
                        )
                    )
                )
                inputs[in_url]["sds"].remove(sd)
            else:
                sd["start_time_dimensions"] = sd["time_dimensions"][
                    requested_time_dimensions
                ]
                sd["end_time_dimensions"] = end_times[requested_time_dimensions]
                sd["requested_time_dimensions"] = np.where(requested_time_dimensions)[0]

    if options["print"] in ["grass", "extended"]:
        # ["|".join([[s["url"], s["id"]] + s["extended_metadata"].values])
        # \n".join(
        print_type = "{}_metadata".format(options["print"])
        print(
            sep.join(
                ["id", "url", "rastercount", "time_dimensions"]
                + list(next(iter(inputs.values()))["sds"][0][print_type].keys())
            )
        )
        print(
            "\n".join(
                [
                    sep.join(
                        [
                            sd["id"],
                            sd["url"],
                            str(sd["rastercount"]),
                            str(len(sd["time_dimensions"])),
                        ]
                        + list(map(str, sd[print_type].values()))
                    )
                    for sd in chain.from_iterable([i["sds"] for i in inputs.values()])
                ]
            )
        )
        sys.exit(0)

    # Check if projections match
    if flags["o"]:
        for i in inputs:
            inputs[i]["proj_match"] = True
    else:
        print([i for i in inputs])
        with Pool(processes=int(options["nprocs"])) as pool:
            # Check (only first subdataset) if projections match
            projection_match = pool.map(
                check_projection_match, [i["sds"][0]["url"] for i in inputs]
            )

        for idx, url in enumerate(inputs):
            inputs[url]["proj_match"] = projection_match[idx]

    for sds in inputs.values():

        # Here loop over subdatasets ()
        for sd in sds["sds"]:

            strds_name = (
                "{}_{}".format(options["output"], sd["id"])
                if sd["id"] and not bandref
                else options["output"]
            )

            # Append if exists and overwrite allowed (do not update metadata)
            if strds_name not in existing_strds or (
                gscript.overwrite and not flags["a"]
            ):
                if strds_name not in modified_strds:
                    tgis.open_new_stds(
                        strds_name,
                        "strds",  # type
                        "absolute",  # temporaltype
                        sd["grass_metadata"]["title"],
                        sd["grass_metadata"]["description"],
                        "mean",  # semanticstype
                        None,  # dbif
                        gscript.overwrite,
                    )
                    modified_strds[strds_name] = []
            elif not flags["a"]:
                gscript.fatal(_("STRDS exisits."))

            else:
                modified_strds[strds_name] = []

            if strds_name not in existing_strds:
                existing_strds.append(strds_name)

            import_module, resample = get_import_type(
                sd["url"], sds["proj_match"], options["resample"], flags
            )
            queueing_input.append(
                (
                    sd["url"],
                    sd["rastercount"],
                    sd["grass_metadata"],
                    options,
                    import_module,
                    flags,
                    sd["start_time_dimensions"],
                    sd["end_time_dimensions"],
                    sd["requested_time_dimensions"],
                    grass_env,
                    sds["proj_match"],
                    resample,
                    nodata,
                    strds_name,
                )
            )

    # This is a time consuming part due to building of VRT files
    with Pool(processes=int(options["nprocs"])) as pool:
        queueing_results = pool.starmap(read_data, queueing_input)
    for qres in queueing_results:
        modified_strds[qres[0]].extend(qres[1])
        queued_modules.extend(qres[2])

    queue = ParallelModuleQueue(nprocs=options["nprocs"])
    for mm in queued_modules:
        queue.put(
            MultiModule(
                module_list=mm,
                sync=True,
                set_temp_region=False,
            )
        )
    queue.wait()

    for strds_name, r_maps in modified_strds.items():
        # Register raster maps in strds using tgis
        tgis_strds = tgis.SpaceTimeRasterDataset(strds_name + "@" + grass_env["MAPSET"])
        if grass_version >= [8, 1]:
            map_file = StringIO("\n".join(r_maps))
        else:
            map_file = gscript.tempfile()
            with open(map_file, "w") as mf:
                mf.write("\n".join(r_maps))
        register_maps_in_space_time_dataset(
            "raster",
            strds_name + "@" + grass_env["MAPSET"],
            file=StringIO("\n".join(r_maps)),
            update_cmd_list=False,
        )

        tgis_strds.update_from_registered_maps(dbif=None)

    return 0


if __name__ == "__main__":
    options, flags = gscript.parser()
    sys.exit(main())
