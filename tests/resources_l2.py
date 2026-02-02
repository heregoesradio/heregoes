from tests import input_dir

# L2 CMIPs to test against the original L1b test files for heregoes v0.x

# abi
abi_mc01_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C01_G16_s20211691942252_e20211691942310_c20211691942366.nc"
)
abi_mc02_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C02_G16_s20211691942252_e20211691942310_c20211691942370.nc"
)
abi_mc03_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C03_G16_s20211691942252_e20211691942310_c20211691942377.nc"
)
abi_mc04_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C04_G16_s20211691942252_e20211691942310_c20211691942365.nc"
)
abi_mc05_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C05_G16_s20211691942252_e20211691942310_c20211691942376.nc"
)
abi_mc06_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C06_G16_s20211691942252_e20211691942315_c20211691942365.nc"
)
abi_mc07_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C07_G16_s20211691942252_e20211691942321_c20211691942386.nc"
)
abi_mc08_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C08_G16_s20211691942252_e20211691942310_c20211691942386.nc"
)
abi_mc09_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C09_G16_s20211691942252_e20211691942315_c20211691942387.nc"
)
abi_mc10_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C10_G16_s20211691942252_e20211691942322_c20211691942370.nc"
)
abi_mc11_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C11_G16_s20211691942252_e20211691942310_c20211691942376.nc"
)
abi_mc12_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C12_G16_s20211691942252_e20211691942316_c20211691942377.nc"
)
abi_mc13_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C13_G16_s20211691942252_e20211691942322_c20211691942387.nc"
)
abi_mc14_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C14_G16_s20211691942252_e20211691942310_c20211691942386.nc"
)
abi_mc15_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C15_G16_s20211691942252_e20211691942316_c20211691942386.nc"
)
abi_mc16_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPM1-M6C16_G16_s20211691942252_e20211691942322_c20211691942399.nc"
)

# add a few g16 conuses to test off-earth pixels
abi_cc01_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPC-M6C01_G16_s20211691941174_e20211691943547_c20211691944027.nc"
)
abi_cc02_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPC-M6C02_G16_s20211691941174_e20211691943547_c20211691944046.nc"
)
abi_cc03_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPC-M6C03_G16_s20211691941174_e20211691943547_c20211691944027.nc"
)
abi_cc07_nc = input_dir.joinpath(
    "abi-l2/OR_ABI-L2-CMIPC-M6C07_G16_s20211691941174_e20211691943558_c20211691944032.nc"
)

conus_ncs = [
    abi_cc01_nc,
    abi_cc02_nc,
    abi_cc03_nc,
    abi_cc07_nc,
]

meso_ncs = [
    abi_mc01_nc,
    abi_mc02_nc,
    abi_mc03_nc,
    abi_mc04_nc,
    abi_mc05_nc,
    abi_mc06_nc,
    abi_mc07_nc,
    abi_mc08_nc,
    abi_mc09_nc,
    abi_mc10_nc,
    abi_mc11_nc,
    abi_mc12_nc,
    abi_mc13_nc,
    abi_mc14_nc,
    abi_mc15_nc,
    abi_mc16_nc,
]

abi_ncs = conus_ncs + meso_ncs
