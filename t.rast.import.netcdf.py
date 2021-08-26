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
#%end

#%flag
#% key: p
#% description: Print file structure and exit
#%end

#%flag
#% key: r
#% description: Import only within current region
#%end

#%flag
#% key: l
#% description: Link the raster files using r.external
#%end

#%flag
#% key: e
#% description: Extend location extents based on new dataset
#%end

#%flag
#% key: o
#% label: Override projection check (use current location's projection)
#% description: Assume that the dataset has same projection as the current location
#%end

#%option G_OPT_F_INPUT
#% key: input
#% type: string
#% required: yes
#% multiple: no
#% key_desc: Input file(s) ("-" = stdin)
#% description: URL or name of input netcdf-file ("-" = stdin)
#%end

#%option G_OPT_STRDS_OUTPUT
#% required: yes
#% multiple: no
#% description: Name of the output space time raster dataset
#%end

#%option G_OPT_R_BASENAME_OUTPUT
#% key: basename
#% type: string
#% required: no
#% multiple: no
#% label: Basename of the new generated output maps
#% description: A numerical suffix or timestamp separated by an underscore will be attached to create a unique identifier
#%end

#%option
#% key: title
#% type: string
#% required: no
#% multiple: no
#% description: Title of the new space time dataset
#%end

#%option
#% key: description
#% type: string
#% required: no
#% multiple: no
#% description: Description of the new space time dataset
#%end

#%option
#% key: subdatasets
#% type: string
#% required: no
#% multiple: no
#% description: Regular expression to filter subdatasets
#%end

#%option
#% key: resample
#% type: string
#% required: no
#% multiple: no
#% label: Resampling method when data is reprojected
#% options: nearest, bilinear, bilinear_f, bicubic, bicubic_f, cubicspline, lanczos, lanczos_f, min, Q1, average, med, Q3, max, mode
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
#%end

#%option
#% key: nprocs
#% type: integer
#% required: no
#% multiple: no
#% key_desc: Number of cores
#% label: Number of cores to use during import
#% answer: 1
#%end

#%option G_OPT_F_INPUT
#% key: bandref
#% type: string
#% required: no
#% multiple: no
#% key_desc: Input file with bandreference configuration ("-" = stdin)
#% description: File with mapping of variables or subdatasets to band references
#%end

# Todo:
# - Allow user to choose what metadata to print
#   (e.g. no data values, CF-tags, scaling, cloud coverage)
#   maybe add some of it in the map name, so STRDS could be filtered based on that
# - Improve parallel import (does not work for single layer subdatasets)
# - Make use of more metadata (units, scaling)
# - Add rules to options / flags
# - add testsuite

import sys
from copy import deepcopy
from subprocess import PIPE
from pathlib import Path
import re
import os

import numpy as np

# import dateutil.parser as parser

from osgeo import gdal
import cf_units

import grass.script as gscript
import grass.temporal as tgis
from grass.pygrass.modules import Module, MultiModule, ParallelModuleQueue

# from grass.temporal. import update_from_registered_maps
from grass.temporal.register import register_maps_in_space_time_dataset
from grass.temporal.datetime_math import datetime_to_grass_datetime_string

# python3 t.rast.import.netcdf.py -l input=https://thredds.met.no/thredds/fileServer/ngcd/version_21.03/TG/type2/2020/12/NGCD_TG_type2_20201231.nc output=test basename=testt title=test description=test
"""
options = {
    "input": "https://thredds.met.no/thredds/fileServer/ngcd/version_21.03/TG/type2/2020/12/NGCD_TG_type2_20201231.nc",  # "https://thredds.met.no/thredds/dodsC/ngcd/version_21.03/TG/type2/2020/12/NGCD_TG_type2_20201231.nc", #"https://thredds.met.no/thredds/catalog/senorge/seNorge2_1/TEMP1d/catalog.html?dataset=senorge/seNorge2_1/TEMP1d/seNorge_v2_1_TEMP1d_grid_2015.nc",
    "output": "test",  # Name of the output space time raster dataset
    "basename": "test",  # Basename of the new generated output maps
    "title": "test",  # Title of the new space time dataset
    "description": "test",  # Description of the new space time dataset
    "subdatasets": "*",  # Regular expression for filtering subdatasets
    "memory": 300,
}

flags = {
    "a": False,  # Append to existing STRDS
    "r": False,  # Set the current region from the last map that was imported
    "l": False,  # Link the raster files using r.external
    "e": False,  # Extend location extents based on new dataset
    "o": True,  # Override projection check (use current location's projection)
}
input_y = "https://thredds.met.no/thredds/fileServer/senorge/seNorge2_1/TEMP1d/seNorge_v2_1_TEMP1d_grid_2015.nc"
input = "https://thredds.met.no/thredds/fileServer/ngcd/version_21.03/TG/type2/2020/12/NGCD_TG_type2_20201231.nc"
input_s = "https://nbstds.met.no/thredds/fileServer/NBS/S2A/2021/05/15/S2A_MSIL1C_20210518T105621_NO300_R094_T32VMN_20210512T131440.nc"
# "https://thredds.met.no/thredds/dodsC/ngcd/version_21.03/TG/type2/2020/12/NGCD_TG_type2_20201231.nc"
"""
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


def legalize_name_string(string):
    """"""
    legal_string = re.sub(r"[^\w\d-]+|[^\x00-\x7F]+|[ -/\\]+", "_", string)
    return legal_string


def get_time_dimensions(meta):
    """Extracts netcdf-cf compliant time dimensions from metadata using UDUNITS2"""
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

    if int(gscript.version()["version"].split(".")[0]) < 8:
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
            elif len(line.split("=")) == 2:
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

    """
        "TG#cell_methods": "time: mean within days",
        "TG#standard_name": "air_temperature",
        "TG#long_name": "daily mean temperature",
        "TG#coordinates": "lon lat",
        "TG#grid_mapping": "projection_laea",
        "TG#prod_date": "2021-02-16",
        "TG#software_release": "v0.1.0-beta",
        "TG#_FillValue": "-999.98999",
    """
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


def get_import_type(url, resample, flags_dict, netcdf):
    """Define import type ("r.in.gdal", "r.import", "r.external")"""

    # Check if projections match
    try:
        projection_match = Module(
            "r.external", input=url, flags="j", stderr_=PIPE, stdout_=PIPE
        )
        projection_match = True
    except Exception:
        projection_match = False

    if not projection_match and not flags_dict["o"]:
        if not resample:
            gscript.warning(
                _(
                    "Projections do not match but no resampling method specified. "
                    "Using nearest neighbor method for resampling."
                )
            )
            resample = "nearest"
        if flags_dict["l"]:
            gscript.warning(
                _("Cannot link input directly, using a warped virtual raster")
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
                resample = resample_dict["grass"][resample]

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


# import or link data
def read_data(
    netcdf,
    metadata,
    options_dict,
    import_type,
    ignore_crs,
    crop_to_region,
    time_dimensions,
    gisenv,
    index,
    resamp,
):
    """Import or link data and metadata"""
    maps = []
    imp_flags = "o" if ignore_crs else ""
    # r.external [-feahvtr]
    # r.import [-enl]
    # r.in.gdal [-eflakcrp]
    input_url = netcdf.GetDescription()
    is_subdataset = input_url.startswith("NETCDF")
    # Setup import module
    import_mod = Module(
        import_type,
        input=input_url
        if not is_subdataset
        else create_vrt(netcdf, gisenv, index, resamp),
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
        import_mod.flags.r = crop_to_region
        # import_mod.offset = options["offset"]
        # import_mod.num_digits = options["num_digits"]
    # Setup metadata module
    meta_mod = Module("r.support", run_=False, finish_=False, **metadata)
    # Setup timestamp module
    time_mod = Module("r.timestamp", run_=False, finish_=False)
    # r.timestamp map=soils date='18 feb 2005 10:30:00/20 jul 2007 20:30:00'
    queue = ParallelModuleQueue(nprocs=options_dict["nprocs"])
    mapname_list = [options_dict["basename"]]
    infile = Path(input_url).name.split(":")
    mapname_list.append(legalize_name_string(infile[0]))
    if is_subdataset:
        mapname_list.append(legalize_name_string(infile[1]))
    for i in range(netcdf.RasterCount):
        mapname = "_".join(mapname_list + [time_dimensions[i].strftime("%Y_%m_%d")])
        maps.append(mapname + "@" + gisenv["MAPSET"])
        new_import = deepcopy(import_mod)
        new_import(band=i + 1, output=mapname)
        new_meta = deepcopy(meta_mod)
        new_meta(map=mapname)
        new_time = deepcopy(time_mod)
        new_time(
            map=mapname, date=datetime_to_grass_datetime_string(time_dimensions[i])
        )
        # print(new_import, new_meta, new_time)
        mm = MultiModule(
            module_list=[new_import, new_meta, new_time],
            sync=True,
            set_temp_region=False,
        )
        queue.put(mm)
    queue.wait()
    # print(queue)
    # queue.wait()
    # proc_list = queue.get_finished_modules()
    # print(proc_list)
    # queue.get_num_run_procs()
    return maps


def create_vrt(subdataset, gisenv, index, warp):
    """"""
    vrt_dir = Path(gisenv["GISDBASE"]).joinpath(
        gisenv["LOCATION_NAME"], gisenv["MAPSET"], "gdal"
    )
    sds_name = subdataset.GetDescription()
    if not vrt_dir.is_dir():
        vrt_dir.mkdir()
    vrt_name = str(
        vrt_dir.joinpath(
            "netcdf_{}_{}.vrt".format(index, legalize_name_string(Path(sds_name).name))
        )
    )
    if warp is None or warp == "":
        vrt = gdal.BuildVRT(vrt_name, sds_name)
    else:
        vrt = gdal.Warp(
            vrt_name,
            sds_name,
            options=gdal.WarpOptions(
                format="VRT",
                # outputType=gdal.GDT_Int16,
                # xRes=10,
                # yRes=10,
                dstSRS=gscript.read_command("g.proj", flags="wf").strip(),
                # srcNodata=1,
                resampleAlg="near",
            ),
        )
    vrt = None
    vrt = vrt_name

    return vrt


def main():
    """run the main workflow"""

    input = options["input"].split(",")

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

    # title = options["title"]

    # description = options["description"]

    # resample = options["resample"]

    grass_env = gscript.gisenv()

    tgis.init()

    # Get existing STRDS
    existing_strds = Module(
        "t.list",
        type="strds",
        columns="name",
        where="mapset = '{}'".format(grass_env["MAPSET"]),
        stdout_=PIPE,
    ).outputs.stdout.split("\n")

    # Check inputs
    # URL or file readable
    # STRDS exists / appcreation_dateend to
    # get basename from existing STRDS

    modified_strds = {}

    for in_url in input:

        # Check if file exists and readable
        try:
            ncdf = gdal.Open(in_url)
        except FileNotFoundError:
            gscript.fatal(_("Could not open <{}>".format(in_url)))

        ncdf_metadata = ncdf.GetMetadata()

        cf_version = ncdf_metadata.get("NC_GLOBAL#Conventions")
        # [key for key in metadata.keys() if key in metadata_y.keys()]
        if not cf_version.upper().startswith("CF"):
            gscript.warning(
                _(
                    "Input netCDF file does not adhere to CF-standard. Import may fail or be incorrect."
                )
            )

        # try:
        #     creation_date = parser.parse(ncdf_metadata.get("NC_GLOBAL#creation_date"))
        # except Exception:
        #     creation_date = None

        # Check for subdatasets
        # sds = ncdf.GetSubDatasets()

        # raster_layers = ncdf.RasterCount

        # Sub datasets containing variables have 3 dimensions (x,y,z)
        sds = [
            (sds[0].split(":")[-1], sds[0], len(sds[1].split(" ")[0].split("x")))
            for sds in ncdf.GetSubDatasets()
            if len(sds[1].split(" ")[0].split("x")) == 3
        ]
        if bandref is not None:
            sds = [s for s in sds if s[0] in bandref.keys()]

        if len(sds) > 0 and ncdf.RasterCount == 0:
            open_sds = [gdal.Open(s[1]) for s in sds]
        elif len(sds) == 0 and ncdf.RasterCount == 0:
            gscript.fatal(_("No data to import"))
        else:
            # Check raster layers
            open_sds = [ncdf]
            sds = [("", "", 0)]

        if flags["p"]:
            print("\n".join(["{}|{}".format(in_url, sd[0]) for sd in sds]))
            continue

        # Check global level
        # If sds > 1 loop over sds

        # Here loop over subdatasets

        for idx, sd in enumerate(open_sds):

            sd_metadata = sd.GetMetadata()
            # print(sd_metadata)
            #
            strds_name = (
                "{}_{}".format(options["output"], sds[idx][0])
                if sds[idx][0] and not bandref
                else options["output"]
            )

            time_dimensions = get_time_dimensions(sd_metadata)

            # print(sd_metadata)
            sd_metadata = get_metadata(sd_metadata, sds[idx][0], bandref)

            # Append if exists and overwrite allowed (do not update metadata)
            if strds_name not in existing_strds or (
                gscript.overwrite and not flags["a"]
            ):
                if strds_name not in modified_strds:
                    tgis.open_new_stds(
                        strds_name,
                        "strds",  # type
                        "absolute",  # temporaltype
                        sd_metadata["title"],
                        sd_metadata["description"],
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

            import_module, warp = get_import_type(
                in_url, options["resample"], flags, sd
            )

            modified_strds[strds_name].extend(
                read_data(
                    sd,
                    sd_metadata,
                    options,
                    import_module,
                    flags["o"],
                    flags["r"],
                    time_dimensions,
                    grass_env,
                    idx,
                    warp,
                )
            )

    for strds_name, r_maps in modified_strds.items():
        # Register raster maps in strds using tgis
        # tgis.register.register_map_object_list("strds", [tgis.RasterDataset(rmap + "@" + tgis.get_current_mapset()) for rmap in r_maps], tgis.SpaceTimeRasterDataset(strds + "@" + tgis.get_current_mapset())    )
        tgis_strds = tgis.SpaceTimeRasterDataset(strds_name + "@" + grass_env["MAPSET"])
        print("registering maps", r_maps)
        # register_map_object_list("raster",
        # [tgis.RasterDataset(rmap) for rmap in r_maps], output_stds=tgis_strds, delete_empty=True)

        register_maps_in_space_time_dataset(
            "raster",  # type,
            strds_name + "@" + grass_env["MAPSET"],
            maps=",".join(r_maps),
            update_cmd_list=False,
        )

        tgis_strds.update_from_registered_maps(dbif=None)

    {
        "cell_methods": "time: mean within days",
        "coordinates": "lon lat",
        "grid_mapping": "projection_laea",
        "long_name": "daily mean temperature",
        "NETCDF_DIM_time": "44193.75",
        "NETCDF_VARNAME": "TG",
        "prod_date": "2021-02-16",
        "software_release": "v0.1.0-beta",
        "standard_name": "air_temperature",
        "units": "K",
        "_FillValue": "-999.98999",
    }

    {
        "NC_GLOBAL#comment": "Our open data are licensed under Norwegian Licence for Open Government Data (NLOD) or a Creative Commons Attribution 4.0 International License at your preference. Credit should be given to The Norwegian Meteorological institute, shortened “MET Norway”, as the source of data.",
        "NC_GLOBAL#contact": "http://copernicus-support.ecmwf.int",
        "NC_GLOBAL#Conventions": "CF-1.7",
        "NC_GLOBAL#creation_date": "2021-03-19",
        "NC_GLOBAL#history": "March 2021 creation",
        "NC_GLOBAL#keywords": "Fennoscandia, land, observational gridded dataset, daily mean temperature, past",
        "NC_GLOBAL#license": "https://www.met.no/en/free-meteorological-data/Licensing-and-crediting",
        "NC_GLOBAL#project": "C3S 311a Lot4",
        "TG#cell_methods": "time: mean within days",
        "TG#coordinates": "lon lat",
        "TG#grid_mapping": "projection_laea",
        "TG#long_name": "daily mean temperature",
        "TG#prod_date": "2021-02-16",
        "TG#software_release": "v0.1.0-beta",
        "TG#standard_name": "air_temperature",
        "TG#units": "K",
        "TG#_FillValue": "-999.98999",
        # 'time#axis': 'T',
        # 'time#bounds': 'time_bounds',
        # 'time#calendar': 'standard',
        # 'time#long_name': 'time',
        # 'time#standard_name': 'time',
        # 'time#units': 'days since 1900-01-01 00:00:00',
    }

    return 0


if __name__ == "__main__":
    options, flags = gscript.parser()
    sys.exit(main())
