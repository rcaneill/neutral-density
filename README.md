# Make a surface more neutral, using JAX

## Data we use

We use WOA18 1 degree:
```bash
cd data
wget https://www.ncei.noaa.gov/thredds-ocean/fileServer/woa23/DATA/temperature/netcdf/decav/1.00/woa23_decav_t00_01.nc
wget https://www.ncei.noaa.gov/thredds-ocean/fileServer/woa23/DATA/salinity/netcdf/decav/1.00/woa23_decav_s00_01.nc
```




## How to install and run

You will first need to install `uv`.

start all commands with `uv run --frozen`

```bash
uv run --frozen jupyter lab
uv run --frozen pytest
```

## How to add packages

Run `uv add package-name`. Note that it can only install pip available
packages.
If binary packages are needed, we either install them by hand, or I (Romain)
can start to build containers (apptainer, docker).

## Linting

We use [ruff](https://github.com/astral-sh/ruff) for linting.

You can install [pre-commit](https://pre-commit.com/#install) to run the linting
automatically at each commit.
