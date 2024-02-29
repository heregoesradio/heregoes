from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.resolve()

input_dir = SCRIPT_PATH.joinpath("input")
input_dir.mkdir(parents=True, exist_ok=True)

# abi
abi_mc01_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C01_G16_s20211691942252_e20211691942310_c20211691942342.nc"
)
abi_mc02_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C02_G16_s20211691942252_e20211691942310_c20211691942334.nc"
)
abi_mc03_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C03_G16_s20211691942252_e20211691942310_c20211691942351.nc"
)
abi_mc04_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C04_G16_s20211691942252_e20211691942310_c20211691942340.nc"
)
abi_mc05_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C05_G16_s20211691942252_e20211691942310_c20211691942347.nc"
)
abi_mc06_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C06_G16_s20211691942252_e20211691942315_c20211691942345.nc"
)
abi_mc07_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C07_G16_s20211691942252_e20211691942321_c20211691942355.nc"
)
abi_mc08_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C08_G16_s20211691942252_e20211691942310_c20211691942357.nc"
)
abi_mc09_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C09_G16_s20211691942252_e20211691942315_c20211691942368.nc"
)
abi_mc10_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C10_G16_s20211691942252_e20211691942322_c20211691942353.nc"
)
abi_mc11_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C11_G16_s20211691942252_e20211691942310_c20211691942348.nc"
)
abi_mc12_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C12_G16_s20211691942252_e20211691942316_c20211691942356.nc"
)
abi_mc13_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C13_G16_s20211691942252_e20211691942322_c20211691942361.nc"
)
abi_mc14_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C14_G16_s20211691942252_e20211691942310_c20211691942364.nc"
)
abi_mc15_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C15_G16_s20211691942252_e20211691942316_c20211691942358.nc"
)
abi_mc16_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C16_G16_s20211691942252_e20211691942322_c20211691942366.nc"
)

# add a few g16 conuses to test off-earth pixels
abi_cc01_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadC-M6C01_G16_s20211691941174_e20211691943547_c20211691943589.nc"
)
abi_cc02_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadC-M6C02_G16_s20211691941174_e20211691943547_c20211691943571.nc"
)
abi_cc03_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadC-M6C03_G16_s20211691941174_e20211691943547_c20211691943587.nc"
)
abi_cc07_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadC-M6C07_G16_s20211691941174_e20211691943558_c20211691944002.nc"
)

abi_ncs = [
    abi_cc01_nc,
    abi_cc02_nc,
    abi_cc03_nc,
    abi_cc07_nc,
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

# suvi
suvi_094_nc = input_dir.joinpath(
    "suvi/OR_SUVI-L1b-Fe093_G16_s20203160623501_e20203160623511_c20203160624091.nc"
)
suvi_131_nc = input_dir.joinpath(
    "suvi/OR_SUVI-L1b-Fe131_G16_s20203160623001_e20203160623011_c20203160623196.nc"
)
suvi_171_nc = input_dir.joinpath(
    "suvi/OR_SUVI-L1b-Fe171_G16_s20203160624201_e20203160624211_c20203160624396.nc"
)
suvi_195_nc = input_dir.joinpath(
    "suvi/OR_SUVI-L1b-Fe195_G16_s20203160623301_e20203160623311_c20203160623491.nc"
)
suvi_284_nc = input_dir.joinpath(
    "suvi/OR_SUVI-L1b-Fe284_G16_s20203160624501_e20203160624511_c20203160625090.nc"
)
suvi_304_nc = input_dir.joinpath(
    "suvi/OR_SUVI-L1b-He303_G16_s20203160622501_e20203160622511_c20203160623090.nc"
)
suvi_ncs = [
    suvi_094_nc,
    suvi_131_nc,
    suvi_171_nc,
    suvi_195_nc,
    suvi_284_nc,
    suvi_304_nc,
]
