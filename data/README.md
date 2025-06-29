# Data Directory

## Shapefiles

The US county shapefiles are not included in this repository due to size constraints. 

To download the required county boundary files:

1. Visit the US Census Bureau TIGER/Line Files: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
2. Download the current year's "Counties (and equivalent)" shapefile
3. Extract the files to the `data/shapefiles/` directory

The shapefile should include these files:
- `tl_2024_us_county.shp` (or appropriate year)
- `tl_2024_us_county.shx`
- `tl_2024_us_county.dbf`
- `tl_2024_us_county.prj`
- `tl_2024_us_county.cpg`

Alternatively, you can download directly using:
```bash
cd data/shapefiles/
wget https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip
unzip tl_2024_us_county.zip
rm tl_2024_us_county.zip
```

## Climate Data

NEX-GDDP-CMIP6 climate data files should be organized in a separate directory structure as specified in the configuration.