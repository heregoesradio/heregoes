from tests import input_dir

# multiple hurricanes + storms, plus some clear coastlines and mountains
case_dir = input_dir.joinpath("abi-l1b/cases/2019-09-04")

abi_cc02_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadC-M6C02_G16_s20192471701118_e20192471703491_c20192471703540.nc"
)
abi_cc03_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadC-M6C03_G16_s20192471701118_e20192471703491_c20192471703550.nc"
)
abi_cc01_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadC-M6C01_G16_s20192471701118_e20192471703491_c20192471703563.nc"
)
conus_rgb_ncs = [abi_cc02_nc, abi_cc03_nc, abi_cc01_nc]

abi_fc02_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadF-M6C02_G16_s20192471700150_e20192471709458_c20192471709495.nc"
)
abi_fc03_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadF-M6C03_G16_s20192471700150_e20192471709458_c20192471709526.nc"
)
abi_fc01_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadF-M6C01_G16_s20192471700150_e20192471709458_c20192471709516.nc"
)
fulldisk_rgb_ncs = [abi_fc02_nc, abi_fc03_nc, abi_fc01_nc]

abi_m1c02_T1700_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadM1-M6C02_G16_s20192471700226_e20192471700283_c20192471700322.nc"
)
abi_m1c03_T1700_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadM1-M6C03_G16_s20192471700226_e20192471700283_c20192471700334.nc"
)
abi_m1c01_T1700_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadM1-M6C01_G16_s20192471700226_e20192471700283_c20192471700340.nc"
)
meso1_t1700_rgb_ncs = [abi_m1c02_T1700_nc, abi_m1c03_T1700_nc, abi_m1c01_T1700_nc]

abi_m1c02_T1701_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadM1-M6C02_G16_s20192471701197_e20192471701254_c20192471701291.nc"
)
abi_m1c03_T1701_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadM1-M6C03_G16_s20192471701197_e20192471701254_c20192471701311.nc"
)
abi_m1c01_T1701_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadM1-M6C01_G16_s20192471701197_e20192471701254_c20192471701316.nc"
)
meso1_t1701_rgb_ncs = [abi_m1c02_T1701_nc, abi_m1c03_T1701_nc, abi_m1c01_T1701_nc]

abi_m2c02_T1700_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadM2-M6C02_G16_s20192471700497_e20192471700554_c20192471700590.nc"
)
abi_m2c03_T1700_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadM2-M6C03_G16_s20192471700497_e20192471700554_c20192471701009.nc"
)
abi_m2c01_T1700_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadM2-M6C01_G16_s20192471700497_e20192471700554_c20192471701004.nc"
)
meso2_t1700_rgb_ncs = [abi_m2c02_T1700_nc, abi_m2c03_T1700_nc, abi_m2c01_T1700_nc]

abi_m2c02_T1701_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadM2-M6C02_G16_s20192471701497_e20192471701554_c20192471701591.nc"
)
abi_m2c03_T1701_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadM2-M6C03_G16_s20192471701497_e20192471701554_c20192471702011.nc"
)
abi_m2c01_T1701_nc = case_dir.joinpath(
    "OR_ABI-L1b-RadM2-M6C01_G16_s20192471701497_e20192471701554_c20192471702001.nc"
)
meso2_t1701_rgb_ncs = [abi_m2c02_T1701_nc, abi_m2c03_T1701_nc, abi_m2c01_T1701_nc]
