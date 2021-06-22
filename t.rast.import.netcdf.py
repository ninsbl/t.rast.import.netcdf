#!/usr/bin/env python3
############################################################################
#
# MODULE:       t.rast.import.netcdf
# AUTHOR(S):    stefan.blumentrath
# PURPOSE:      Import netCDF files that adhere to the CF convention as a
#               Space Time Raster Dataset (STRDS)
# COPYRIGHT:    (C) 2020 by stefan.blumentrath, and the GRASS Development Team
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
#% description: Import netCDF files that adhere to the CF convention.
#% keyword: temporal
#% keyword: import
#% keyword: raster
#% keyword: time
#% keyword: netcdf
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
#%option
#% key: input
#% type: string
#% required: yes
#% multiple: no
#% description: URL or name of input netcdf-file
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
#%option G_OPT_R_INTERP_TYPE
#% key: resample
#% type: string
#% required: no
#% multiple: no
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

import sys
from copy import deepcopy
from subprocess import PIPE

import numpy as np

from osgeo import gdal
import cf_units

import grass.script as gscript
import grass.temporal as tgis
from grass.pygrass.modules import Module, MultiModule, ParallelModuleQueue

# from grass.temporal. import update_from_registered_maps
from grass.temporal.register import (
    register_maps_in_space_time_dataset,
    register_map_object_list,
)
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
    "o": False,  # Override projection check (use current location's projection)
}
input_y = "https://thredds.met.no/thredds/fileServer/senorge/seNorge2_1/TEMP1d/seNorge_v2_1_TEMP1d_grid_2015.nc"
input = "https://thredds.met.no/thredds/fileServer/ngcd/version_21.03/TG/type2/2020/12/NGCD_TG_type2_20201231.nc"
input_s = "https://nbstds.met.no/thredds/fileServer/NBS/S2A/2021/05/15/S2A_MSIL1C_20210518T105621_NO300_R094_T32VMN_20210512T131440.nc"
# "https://thredds.met.no/thredds/dodsC/ngcd/version_21.03/TG/type2/2020/12/NGCD_TG_type2_20201231.nc"
"""
# Datasets may or may not contain subdatasets
# Datasets may contain several layers
# r.external registers all bands by default


def get_time_dimensions(meta):
    """Extracts netcdf-cf compliant time dimensions from metadata using UDUNITS2"""
    time_values = np.fromstring(
        meta["NETCDF_DIM_time_VALUES"].strip("{").strip("}"), sep=",", dtype=np.float
    )
    time_dates = cf_units.num2date(
        time_values, meta["time#units"], meta["time#calendar"]
    )
    return time_dates


def get_metadata(netcdf):
    """ """
    # title , history , institution , source , comment and references
    ncdf_metadata = netcdf.GetMetadata()

    meta = {}
    # title is required metadata for netCDF-CF
    title = ncdf_metadata.get("NC_GLOBAL#title")
    title += (
        ", version: {}".format(ncdf_metadata["NC_GLOBAL#version"])
        if "NC_GLOBAL#version" in ncdf_metadata
        else ""
    )
    title += (
        ", type: {}".format(ncdf_metadata["NC_GLOBAL#type"])
        if "NC_GLOBAL#type" in ncdf_metadata
        else ""
    )
    meta["title"] = title
    # history is required metadata for netCDF-CF
    meta["history"] = ncdf_metadata[
        "NC_GLOBAL#history"
    ]  # phrase Text to append to the next line of the map's metadata file
    meta["units"] = None  # string Text to use for map data units
    meta["vdatum"] = None  # string Text to use for map vertical datum
    meta["source1"] = ncdf_metadata["NC_GLOBAL#source"]
    meta["source2"] = ncdf_metadata["NC_GLOBAL#institution"]
    description = ncdf_metadata.get("NC_GLOBAL#summary")
    description += (
        "\n{}".format(ncdf_metadata["NC_GLOBAL#summary"])
        if "NC_GLOBAL#summary" in ncdf_metadata
        else ""
    )
    description += (
        "\n{}".format(ncdf_metadata["NC_GLOBAL#comment"])
        if "NC_GLOBAL#comment" in ncdf_metadata
        else ""
    )
    description += (
        "\n{}".format(ncdf_metadata["NC_GLOBAL#references"])
        if "NC_GLOBAL#references" in ncdf_metadata
        else ""
    )
    meta["description"] = description

    cf_version = ncdf_metadata["NC_GLOBAL#Conventions"]
    # [key for key in metadata.keys() if key in metadata_y.keys()]

    # Check for subdatasets
    sds = [sds[0].split(":")[-1] for sds in netcdf.GetSubDatasets()]

    # ncdf.GetRasterBand(1).GetMetadata()
    return cf_version, meta, sds


def get_import_type(options, flags, netcdf):
    """Define import type ("r.in.gdal", "r.import", "r.external")"""
    # Check projection match
    try:
        projection_match = Module(
            "r.external", input=input, flags="j", stderr_=PIPE, stdout_=PIPE
        )
        projection_match = True
    except Exception:
        projection_match = False
    if flags["l"]:
        if not projection_match and not flags["o"]:
            gscript.fatal(_("Cannot link input, projections do not match"))
        import_type = "r.external"
    elif options["resample"]:
        if not projection_match:
            import_type = "r.import"
        else:
            gscript.warning(_("Projections match, no resampling needed."))
            import_type = "r.in.gdal"
    else:
        if not projection_match and not flags["o"]:
            gscript.fatal(_("Projections do not match"))
        import_type = "r.in.gdal"
    return import_type


# import or link data
def read_data(
    netcdf,
    metadata,
    options,
    import_type,
    ignore_crs,
    crop_to_region,
    time_dimensions,
    input,
):
    """Import or link data"""
    maps = []
    imp_flags = "o" if ignore_crs else None
    # r.external [-feahvtr]
    # r.import [-enl]
    # r.in.gdal [-eflakcrp]

    print(gscript.overwrite())

    # Setup import module
    import_mod = Module(
        import_type,
        input=input,
        run_=False,
        finish_=False,
        flags=imp_flags,
        overwrite=gscript.overwrite(),
    )
    # import_mod.flags.l = ignore_crs
    if import_type != "r.external":
        import_mod.memory = options["memory"]
    if import_type == "r.import":
        import_mod.resample = options["resample"]
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
    queue = ParallelModuleQueue(nprocs=options["nprocs"])
    import_list = []
    for i in range(netcdf.RasterCount):
        mapname = "{}_{}".format(
            options["basename"], time_dimensions[i].strftime("%Y_%m_%d")
        )
        maps.append(mapname + "@" + tgis.get_current_mapset())
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
    print(queue)
    # queue.wait()
    proc_list = queue.get_finished_modules()
    print(proc_list)
    # queue.get_num_run_procs()
    return maps


def main():
    """run the main workflow"""

    input = options["input"]
    input = "/vsicurl/" + input if input.startswith("http") else input

    strds = options["output"]

    title = options["title"]

    description = options["description"]

    resample = options["resample"]

    # Check if file exists and readable
    try:
        ncdf = gdal.Open(input)
    except FileNotFoundError:
        gscript.fatal(_("Could not open <{}>".format(options["input"])))

    # Check inputs
    # URL or file readable
    # STRDS exists / append to
    # get basename from existing STRDS

    cf_version, ncdf_metadata, sds = get_metadata(ncdf)

    tgis.init()

    current_mapset = None
    existing_strds = Module(
        "t.list",
        type="strds",
        columns="name",
        where="mapset = '{}'".format(current_mapset),
        stdout_=PIPE,
    ).outputs.stdout.split("\n")

    # Append if exists and overwrite allowed (do not update metadata)
    if not strds in existing_strds or (gscript.overwrite and not append):
        tgis.open_new_stds(
            strds,
            "strds",  # type
            "absolute",  # temporaltype
            title,
            description,
            "mean",  # semanticstype
            None,  # dbif
            gscript.overwrite,
        )
    elif append:
        do_append = True
    else:
        gscript.fatal(_("STRDS exisits."))

    import_module = get_import_type(options, flags, ncdf)

    print(import_module)

    time_dimensions = get_time_dimensions(ncdf.GetMetadata())

    r_maps = read_data(
        ncdf,
        ncdf_metadata,
        options,
        import_module,
        flags["o"],
        flags["r"],
        time_dimensions,
        input,
    )

    # Register in strds using tgis
    print(r_maps)
    # tgis.register.register_map_object_list("strds", [tgis.RasterDataset(rmap + "@" + tgis.get_current_mapset()) for rmap in r_maps], tgis.SpaceTimeRasterDataset(strds + "@" + tgis.get_current_mapset())    )

    tgis_strds = tgis.SpaceTimeRasterDataset(strds + "@" + tgis.get_current_mapset())

    # register_map_object_list("raster",
    # [tgis.RasterDataset(rmap) for rmap in r_maps], output_stds=tgis_strds, delete_empty=True)

    register_maps_in_space_time_dataset(
        "raster",  # type,
        strds + "@" + tgis.get_current_mapset(),
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
