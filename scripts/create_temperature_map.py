#!/usr/bin/env python
"""Create a choropleth map of CONUS temperature data by county."""

import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

print("Loading temperature data...")

# Load the temperature statistics
df = pd.read_csv('climate_outputs/stats/tas/conus/ssp370/conus_ssp370_tas_stats_threshold0.0c.csv')

# Load the county shapefile
gdf = gpd.read_file('regional_counties/conus_counties.shp')

# Check columns in temperature data
print("Temperature data columns:", df.columns.tolist())

# Calculate average temperature metrics per county across all years
print("Calculating county temperature averages...")
county_avg = df.groupby('county_id').agg({
    'mean_annual_temp_c': 'mean',
    'min_temp_c': 'mean',
    'max_temp_c': 'mean',
    'temp_range_c': 'mean',
    'days_above_30c': 'mean',
    'county_name': 'first',
    'state': 'first'
}).reset_index()

# Convert Celsius to Fahrenheit for US audience
county_avg['mean_temp_fahrenheit'] = county_avg['mean_annual_temp_c'] * 9/5 + 32
county_avg['min_temp_fahrenheit'] = county_avg['min_temp_c'] * 9/5 + 32
county_avg['max_temp_fahrenheit'] = county_avg['max_temp_c'] * 9/5 + 32

# Ensure county IDs match format (pad with zeros)
county_avg['GEOID'] = county_avg['county_id'].astype(str).str.zfill(5)

# Merge with shapefile
print("Merging with shapefile...")
map_data = gdf.merge(county_avg, on='GEOID', how='left')

# Create temperature color map (blue to red)
temp_cmap = LinearSegmentedColormap.from_list('temperature',
    ['#313695', '#4575B4', '#74ADD1', '#ABD9E9', '#E0F3F8', 
     '#FEE090', '#FDAE61', '#F46D43', '#D73027', '#A50026'])

# Create figure with single map
fig, ax = plt.subplots(1, 1, figsize=(16, 10))

# Map: Mean Annual Temperature
map_data.plot(column='mean_temp_fahrenheit',
              ax=ax,
              cmap=temp_cmap,
              legend=True,
              legend_kwds={'label': 'Mean Annual Temperature (°F)',
                          'shrink': 0.7},
              edgecolor='#666666',
              linewidth=0.2,
              missing_kwds={'color': 'lightgrey'})

ax.set_title('CONUS Average Annual Temperature by County\n(SSP370 Scenario, 2015-2100)', 
             fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

# Add regional temperature annotations
ax.annotate('Cooler\nNorth', xy=(-100, 48), fontsize=12, fontweight='bold',
            color='#4575B4', ha='center')
ax.annotate('Warmer\nSouth', xy=(-100, 30), fontsize=12, fontweight='bold',
            color='#D73027', ha='center')

# Add statistics text box
stats_text = f"""
Temperature Statistics:
• Min: {map_data['mean_temp_fahrenheit'].min():.1f}°F ({map_data['mean_annual_temp_c'].min():.1f}°C)
• Max: {map_data['mean_temp_fahrenheit'].max():.1f}°F ({map_data['mean_annual_temp_c'].max():.1f}°C)
• Mean: {map_data['mean_temp_fahrenheit'].mean():.1f}°F ({map_data['mean_annual_temp_c'].mean():.1f}°C)
• Counties: {len(map_data[map_data['mean_temp_fahrenheit'].notna()])}
"""

ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save the figure
output_file = 'climate_outputs/conus_temperature_map.png'
print(f"Saving map to {output_file}...")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Temperature map saved successfully!")

# Create a second map showing temperature range
fig2, ax2 = plt.subplots(1, 1, figsize=(16, 10))

# Map: Temperature Range (difference between max and min)
map_data.plot(column='temp_range_c',
              ax=ax2,
              cmap='YlOrRd',
              legend=True,
              legend_kwds={'label': 'Annual Temperature Range (°C)',
                          'shrink': 0.7},
              edgecolor='#666666',
              linewidth=0.2,
              missing_kwds={'color': 'lightgrey'})

ax2.set_title('CONUS Annual Temperature Range by County\n(SSP370 Scenario, 2015-2100 Average)', 
              fontsize=16, fontweight='bold', pad=20)
ax2.axis('off')

# Add annotations
ax2.annotate('More Stable\nTemperatures', xy=(-80, 35), fontsize=12, fontweight='bold',
             color='#FFFFB2', ha='center')
ax2.annotate('Greater\nVariability', xy=(-100, 42), fontsize=12, fontweight='bold',
             color='#BD0026', ha='center')

output_file2 = 'climate_outputs/conus_temperature_range_map.png'
print(f"Saving temperature range map to {output_file2}...")
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"Temperature range map saved successfully!")

# Print summary statistics
print("\n=== Temperature Summary ===")
print(f"Average Temperature: {county_avg['mean_annual_temp_c'].mean():.2f}°C ({county_avg['mean_temp_fahrenheit'].mean():.2f}°F)")
print(f"Coldest County: {county_avg.loc[county_avg['mean_annual_temp_c'].idxmin(), 'county_name']}, {county_avg.loc[county_avg['mean_annual_temp_c'].idxmin(), 'state']}")
print(f"  Temperature: {county_avg['mean_annual_temp_c'].min():.2f}°C ({county_avg['mean_temp_fahrenheit'].min():.2f}°F)")
print(f"Warmest County: {county_avg.loc[county_avg['mean_annual_temp_c'].idxmax(), 'county_name']}, {county_avg.loc[county_avg['mean_annual_temp_c'].idxmax(), 'state']}")
print(f"  Temperature: {county_avg['mean_annual_temp_c'].max():.2f}°C ({county_avg['mean_temp_fahrenheit'].max():.2f}°F)")