from tests import input_dir

# one month of UW CIMSS' Baseline Fit Infrared Emissivity Database (IREMIS)(https://cimss.ssec.wisc.edu/iremis/)
iremis_nc = input_dir.joinpath(
    "ancillary/global_emis_inf10_monthFilled_MYD11C3.A2016183.041.nc"
)
iremis_locations_nc = input_dir.joinpath("ancillary/global_emis_inf10_location.nc")
