#!/usr/bin/env python3

"""
MODULE:    Test of t.rast.import.netcdf

AUTHOR(S): Stefan Blumentrath <stefan dot blumentrath at nina dot no>

PURPOSE:   Test of t.rast.import.netcdf (example of a simple test of a module)

COPYRIGHT: (C) 2021 by Stefan Blumentrath and the GRASS Development Team

This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.
"""

import grass.script as gs

from grass.gunittest.case import TestCase
from grass.gunittest.main import test


def get_raster_min_max(raster_map):
    info = gs.raster_info(raster_map)
    return info["min"], info["max"]


class TestWatershed(TestCase):
    """The main (and only) test case for the t.rast.import.netcdf module"""

    # NetCDF URL to be used as input for sentinel data test
    input_sentinel = ["https://nbstds.met.no/thredds/fileServer/NBS/S2A/2021/02/28/S2A_MSIL1C_20210228T103021_N0202_R108_T35WPU_20210228T201033_DTERRENGDATA.nc",
    "https://nbstds.met.no/thredds/fileServer/NBS/S2A/2021/02/28/S2A_MSIL1C_20210228T103021_N0202_R108_T32VNL_20210228T201033_DTERRENGDATA.nc"]
    # STRDS to be used as output for sentinel data test
    output_sentinel = "S2"
    # NetCDF URL to be used as input for climate data test
    input_climate = "https://thredds.met.no/thredds/fileServer/senorge/seNorge_1957/Archive/seNorge2018_2021.nc"
    # Input file name
    input_file = "url_list.txt"
    # STRDS to be used as output for climate data test
    output_climate = "se_norge"

    @classmethod
    def setUpClass(cls):
        """Ensures expected computational region (and anything else needed)

        These are things needed by all test function but not modified by
        any of them.
        """
        # We will use specific computational region for our process in case
        # something else is running in parallel with our tests.
        cls.use_temp_region()
        # Use of of the inputs to set computational region
        cls.runModule("g.region", raster=cls.test_input_1)

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary region (and anything else we created)"""
        cls.del_temp_region()

    def tearDown(self):
        """Remove the output created from the module

        This is executed after each test function run. If we had
        something to set up before each test function run, we would use setUp()
        function.

        Since we remove the raster map after running each test function,
        we can reuse the same name for all the test functions.
        """
        for strds in [self.output_climate, self.output_sentinel]:
            try:
                self.runModule("t.remove", flags="rf", name=strds)
            except Exception:
                pass

    def test_sentinel_output_created(self):
        """Check that the output is created"""
        # run the import module
        self.assertModule(
            "t.rast.import.netcdf",
            flags="lo",
            input=self.input_sentinel[0],
            output=self.output_sentinel,
            bandref="bandref_2.conf",
            memory=2048,
            nprocs=2,
        )
        # check to see if output is in mapset
        # Adjust to STRDS
        # self.assertRasterExists(self.output, msg="Output was not created")

    def test_sentinel_output_appended(self):
        """Check that the output is created"""
        # run the import module
        self.assertModule(
            "t.rast.import.netcdf",
            flags="lo",
            input=self.input_sentinel[0],
            output=self.output_sentinel,
            bandref="bandref_2.conf",
            memory=2048,
            nprocs=2,
        )
        self.assertModule(
            "t.rast.import.netcdf",
            flags="loa",
            input=self.input_sentinel[1],
            output=self.output_sentinel,
            bandref="bandref_2.conf",
            memory=2048,
            nprocs=2,
        )

    def test_sentinel_input_comma_separated(self):
        """Check that the output is created"""
        self.assertModule(
            "t.rast.import.netcdf",
            flags="lo",
            input=",".join(self.input_sentinel),
            output=self.output_sentinel,
            bandref="bandref_2.conf",
            memory=2048,
            nprocs=2,
        )

    def test_sentinel_input_file(self):
        """Check that the output is created"""
        with open(input_file, "w") as f:
            f.write("\n".join(self.input_sentinel))
        self.assertModule(
            "t.rast.import.netcdf",
            flags="lo",
            input=self.input_file,
            output=self.output_sentinel,
            bandref="bandref_2.conf",
            memory=2048,
            nprocs=2,
        )

<h3>Append to STRDS from previous imports</h3>
<div class="code"><pre>
# Choose dataset to import (see also m.crawl.thredds module)

# Create a band reference configuraton file
echo "tg=temperature_avg
tn=temperature_min" > bandref.conf

# Import data
t.rast.import.netcdf output=SeNorge bandref=bandref.conf memory=2048 nprocs=2 -a \
input=https://thredds.met.no/thredds/fileServer/senorge/seNorge_2020/Archive/seNorge2018_2021.nc \
</pre></div>

# Create a band reference configuraton file
echo "tg=temperature_avg
tn=temperature_min" > bandref.conf

    def test_climate_output_created(self):
        """Check that the output is created"""
        # run the watershed module
        self.assertModule(
            "t.rast.import.netcdf",
            flags="lo",
            input=self.input_climate,
            output=self.output_climate,
            bandref="bandref_sn.conf",
            memory=2048,
            nprocs=2,
        )



    def test_missing_parameter(self):
        """Check that the module fails when parameters are missing

        Checks absence of each of the three parameters. Each parameter absence
        is tested separatelly.

        Note that this does not cover all the possible combinations, but it
        tries to simulate most of possible user errors and it should cover
        most of the implementation.
        """
        self.assertModuleFail(
            "t.rast.import.netcdf",
            b_input=self.test_input_2,
            output=self.output,
            msg="The a_input parameter should be required",
        )
        self.assertModuleFail(
            "t.rast.import.netcdf",
            a_input=self.test_input_1,
            output=self.output,
            msg="The b_input parameter should be required",
        )
        self.assertModuleFail(
            "t.rast.import.netcdf",
            a_input=self.test_input_1,
            b_input=self.test_input_2,
            msg="The output parameter should be required",
        )

    def test_output_range(self):
        """Check to see if output is within the expected range"""
        self.assertModule(
            "t.rast.import.netcdf",
            a_input=self.test_input_1,
            b_input=self.test_input_2,
            output=self.output,
        )

        min_1, max_1 = get_raster_min_max(self.test_input_1)
        min_2, max_2 = get_raster_min_max(self.test_input_2)

        reference_min = min_1 + min_2
        reference_max = max_1 + max_2

        self.assertRasterMinMax(
            self.output,
            reference_min,
            reference_max,
            msg="Output exceeds the values computed from inputs",
        )



<h3>Append to STRDS from previous imports</h3>
<div class="code"><pre>
# Choose dataset to import (see also m.crawl.thredds module)

# Create a band reference configuraton file
echo "tg=temperature_avg
tn=temperature_min" > bandref.conf

# Import data
t.rast.import.netcdf output=SeNorge bandref=bandref.conf memory=2048 nprocs=2 -a \
input=https://thredds.met.no/thredds/fileServer/senorge/seNorge_2020/Archive/seNorge2018_2021.nc \
</pre></div>


if __name__ == "__main__":
    test()
