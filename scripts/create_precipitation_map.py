#!/usr/bin/env python
"""Create a choropleth map of CONUS precipitation data by county."""

import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

print("Loading data...")

# Load the precipitation statistics
df = pd.read_csv('climate_outputs/stats/pr/conus/ssp370/conus_ssp370_pr_stats_threshold25.4mm.csv')

# Load the county shapefile
gdf = gpd.read_file('regional_counties/conus_counties.shp')

# Calculate average annual precipitation per county across all years
print("Calculating county averages...")
county_avg = df.groupby('county_id').agg({
    'total_annual_precip_mm': 'mean',
    'days_above_threshold': 'mean',
    'mean_daily_precip_mm': 'mean',
    'county_name': 'first',
    'state': 'first'
}).reset_index()

# Ensure county IDs match format (pad with zeros)
county_avg['GEOID'] = county_avg['county_id'].astype(str).str.zfill(5)

# Merge with shapefile
print("Merging with shapefile...")
map_data = gdf.merge(county_avg, on='GEOID', how='left')

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('CONUS Precipitation Analysis by County (SSP370, 2015-2100 Average)', fontsize=16, fontweight='bold')

# Color schemes
precip_cmap = LinearSegmentedColormap.from_list('precipitation', 
    ['#FFFFCC', '#C7E9B4', '#7FCDBB', '#41B6C4', '#1D91C0', '#225EA8', '#0C2C84'])
threshold_cmap = LinearSegmentedColormap.from_list('threshold', 
    ['#FFF7EC', '#FEE8C8', '#FDD49E', '#FDBB84', '#FC8D59', '#EF6548', '#D7301F', '#990000'])

# Map 1: Total Annual Precipitation
ax1 = axes[0, 0]
map_data.plot(column='total_annual_precip_mm', 
              ax=ax1,
              cmap=precip_cmap,
              legend=True,
              legend_kwds={'label': 'Precipitation (mm/year)', 
                          'orientation': 'horizontal',
                          'pad': 0.03,
                          'fraction': 0.05},
              edgecolor='#333333',
              linewidth=0.1,
              missing_kwds={'color': 'lightgrey'})
ax1.set_title('Average Annual Precipitation', fontsize=14, fontweight='bold')
ax1.axis('off')

# Map 2: Days Above Threshold (Heavy Rain Days)
ax2 = axes[0, 1]
map_data.plot(column='days_above_threshold',
              ax=ax2,
              cmap=threshold_cmap,
              legend=True,
              legend_kwds={'label': 'Days > 25.4mm (1 inch)',
                          'orientation': 'horizontal',
                          'pad': 0.03,
                          'fraction': 0.05},
              edgecolor='#333333',
              linewidth=0.1,
              missing_kwds={'color': 'lightgrey'})
ax2.set_title('Average Days with Heavy Rainfall (>1 inch)', fontsize=14, fontweight='bold')
ax2.axis('off')

# Map 3: Mean Daily Precipitation
ax3 = axes[1, 0]
map_data.plot(column='mean_daily_precip_mm',
              ax=ax3,
              cmap=precip_cmap,
              legend=True,
              legend_kwds={'label': 'Precipitation (mm/day)',
                          'orientation': 'horizontal',
                          'pad': 0.03,
                          'fraction': 0.05},
              edgecolor='#333333',
              linewidth=0.1,
              missing_kwds={'color': 'lightgrey'})
ax3.set_title('Average Daily Precipitation', fontsize=14, fontweight='bold')
ax3.axis('off')

# Map 4: Precipitation Intensity Index (ratio of total to rainy days)
# Calculate precipitation intensity
map_data['precip_intensity'] = map_data['total_annual_precip_mm'] / (365 - df.groupby('county_id')['dry_days'].mean().values[0])

ax4 = axes[1, 1]
map_data.plot(column='precip_intensity',
              ax=ax4,
              cmap='YlOrRd',
              legend=True,
              legend_kwds={'label': 'Intensity (mm/wet day)',
                          'orientation': 'horizontal',
                          'pad': 0.03,
                          'fraction': 0.05},
              edgecolor='#333333',
              linewidth=0.1,
              missing_kwds={'color': 'lightgrey'})
ax4.set_title('Precipitation Intensity (mm per wet day)', fontsize=14, fontweight='bold')
ax4.axis('off')

# Add statistics text box
stats_text = f"""
Dataset Statistics:
• Counties: {len(map_data)}
• Years: 2015-2100 (86 years)
• Scenario: SSP370
• Min Annual Precip: {map_data['total_annual_precip_mm'].min():.0f} mm
• Max Annual Precip: {map_data['total_annual_precip_mm'].max():.0f} mm
• Mean Annual Precip: {map_data['total_annual_precip_mm'].mean():.0f} mm
"""

fig.text(0.02, 0.02, stats_text, fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save the figure
output_file = 'climate_outputs/conus_precipitation_maps.png'
print(f"Saving map to {output_file}...")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Map saved successfully!")

# Also create a simple single map
fig2, ax = plt.subplots(1, 1, figsize=(16, 10))
map_data.plot(column='total_annual_precip_mm',
              ax=ax,
              cmap=precip_cmap,
              legend=True,
              legend_kwds={'label': 'Average Annual Precipitation (mm)',
                          'shrink': 0.7},
              edgecolor='#666666',
              linewidth=0.2,
              missing_kwds={'color': 'lightgrey'})

ax.set_title('CONUS Average Annual Precipitation by County\n(SSP370 Scenario, 2015-2100)', 
             fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

# Add gradient from dry (west) to wet (east) annotation
ax.annotate('Drier\nRegions', xy=(-120, 38), fontsize=12, fontweight='bold',
            color='#8B4513', ha='center')
ax.annotate('Wetter\nRegions', xy=(-75, 38), fontsize=12, fontweight='bold',
            color='#0C2C84', ha='center')

output_file2 = 'climate_outputs/conus_precipitation_simple_map.png'
print(f"Saving simple map to {output_file2}...")
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"Simple map saved successfully!")

# Don't show plot in non-interactive mode
# plt.show()