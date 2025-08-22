#!/usr/bin/env python3
"""
Visualize Missing Counties from Climate Data Processing

Creates a map showing which US counties have climate data (green) 
and which are missing (red) from the processed dataset.
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_missing_counties_map():
    """Create visualization of missing counties."""
    
    print("Loading data...")
    
    # Load the full US county shapefile
    counties_shp = gpd.read_file('tl_2024_us_county/tl_2024_us_county.shp')
    counties_shp['GEOID_int'] = counties_shp['GEOID'].astype(int)
    
    # Load processed counties from transformed data
    processed_df = pd.read_csv('climate_outputs/transformed/transformed_climate_stats.csv')
    processed_counties = set(processed_df['cid2'].unique())
    
    # Mark counties as processed or missing
    counties_shp['has_data'] = counties_shp['GEOID_int'].isin(processed_counties)
    counties_shp['status'] = counties_shp['has_data'].map({True: 'Has Climate Data', False: 'Missing Data'})
    
    # Calculate statistics
    total_counties = len(counties_shp)
    missing_counties = (~counties_shp['has_data']).sum()
    processed_counties_count = counties_shp['has_data'].sum()
    
    print(f"\nStatistics:")
    print(f"Total counties: {total_counties}")
    print(f"Processed: {processed_counties_count} ({processed_counties_count/total_counties*100:.1f}%)")
    print(f"Missing: {missing_counties} ({missing_counties/total_counties*100:.1f}%)")
    
    # Create figure with subplots for different regions
    fig = plt.figure(figsize=(20, 12))
    
    # Main map - Continental US
    ax_main = plt.subplot(2, 3, (1, 4))
    
    # Filter for continental US (exclude Alaska, Hawaii, territories)
    conus = counties_shp[counties_shp['STATEFP'].astype(int) < 60]
    conus = conus[~conus['STATEFP'].isin(['02', '15', '72', '78', '66', '69', '60'])]  # Exclude AK, HI, PR, VI, GU, MP, AS
    
    # Plot CONUS
    conus[conus['has_data']].plot(ax=ax_main, color='lightgreen', edgecolor='gray', linewidth=0.1, label='Has Climate Data')
    conus[~conus['has_data']].plot(ax=ax_main, color='salmon', edgecolor='gray', linewidth=0.1, label='Missing Data')
    
    ax_main.set_title('Continental United States - County Climate Data Coverage', fontsize=14, fontweight='bold')
    ax_main.set_xlabel('Longitude')
    ax_main.set_ylabel('Latitude')
    ax_main.legend(loc='lower left', fontsize=10)
    ax_main.set_xlim(-130, -65)
    ax_main.set_ylim(24, 50)
    
    # Alaska
    ax_ak = plt.subplot(2, 3, 2)
    alaska = counties_shp[counties_shp['STATEFP'] == '02']
    if len(alaska) > 0:
        alaska_has = alaska[alaska['has_data']]
        alaska_missing = alaska[~alaska['has_data']]
        
        if len(alaska_has) > 0:
            alaska_has.plot(ax=ax_ak, color='lightgreen', edgecolor='gray', linewidth=0.2)
        if len(alaska_missing) > 0:
            alaska_missing.plot(ax=ax_ak, color='salmon', edgecolor='gray', linewidth=0.2)
            
        ax_ak.set_title('Alaska', fontsize=12)
        ax_ak.set_xticks([])
        ax_ak.set_yticks([])
        ax_ak.set_aspect('equal')
        
        # Add count
        ak_processed = alaska['has_data'].sum()
        ak_total = len(alaska)
        ax_ak.text(0.5, -0.1, f'{ak_processed}/{ak_total} counties', 
                   transform=ax_ak.transAxes, ha='center', fontsize=9)
    
    # Hawaii
    ax_hi = plt.subplot(2, 3, 3)
    hawaii = counties_shp[counties_shp['STATEFP'] == '15']
    if len(hawaii) > 0:
        hawaii_has = hawaii[hawaii['has_data']]
        hawaii_missing = hawaii[~hawaii['has_data']]
        
        if len(hawaii_has) > 0:
            hawaii_has.plot(ax=ax_hi, color='lightgreen', edgecolor='gray', linewidth=0.2)
        if len(hawaii_missing) > 0:
            hawaii_missing.plot(ax=ax_hi, color='salmon', edgecolor='gray', linewidth=0.2)
            
        ax_hi.set_title('Hawaii', fontsize=12)
        ax_hi.set_xticks([])
        ax_hi.set_yticks([])
        ax_hi.set_aspect('equal')
        
        # Add count
        hi_processed = hawaii['has_data'].sum()
        hi_total = len(hawaii)
        ax_hi.text(0.5, -0.1, f'{hi_processed}/{hi_total} counties', 
                   transform=ax_hi.transAxes, ha='center', fontsize=9)
    
    # Puerto Rico
    ax_pr = plt.subplot(2, 3, 5)
    puerto_rico = counties_shp[counties_shp['STATEFP'] == '72']
    if len(puerto_rico) > 0:
        pr_has = puerto_rico[puerto_rico['has_data']]
        pr_missing = puerto_rico[~puerto_rico['has_data']]
        
        if len(pr_has) > 0:
            pr_has.plot(ax=ax_pr, color='lightgreen', edgecolor='gray', linewidth=0.2)
        if len(pr_missing) > 0:
            pr_missing.plot(ax=ax_pr, color='salmon', edgecolor='gray', linewidth=0.2)
            
        ax_pr.set_title('Puerto Rico', fontsize=12)
        ax_pr.set_xticks([])
        ax_pr.set_yticks([])
        ax_pr.set_aspect('equal')
        
        # Add count
        pr_processed = puerto_rico['has_data'].sum()
        pr_total = len(puerto_rico)
        ax_pr.text(0.5, -0.1, f'{pr_processed}/{pr_total} municipalities', 
                   transform=ax_pr.transAxes, ha='center', fontsize=9)
    
    # Statistics panel
    ax_stats = plt.subplot(2, 3, 6)
    ax_stats.axis('off')
    
    # Create statistics text
    stats_text = f"""
    Coverage Statistics:
    
    Total US Counties: {total_counties:,}
    Processed: {processed_counties_count:,} ({processed_counties_count/total_counties*100:.1f}%)
    Missing: {missing_counties:,} ({missing_counties/total_counties*100:.1f}%)
    
    Most Affected States (missing):
    """
    
    # Calculate missing by state
    missing_by_state = counties_shp[~counties_shp['has_data']].groupby('STATEFP').size().sort_values(ascending=False).head(10)
    state_names = {
        '36': 'New York', '27': 'Minnesota', '42': 'Pennsylvania',
        '53': 'Washington', '55': 'Wisconsin', '51': 'Virginia',
        '26': 'Michigan', '41': 'Oregon', '06': 'California',
        '16': 'Idaho', '72': 'Puerto Rico'
    }
    
    for state_code, count in missing_by_state.items():
        state_name = state_names.get(state_code, f'State {state_code}')
        stats_text += f"\n  • {state_name}: {count}"
    
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, 
                  fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Overall title
    fig.suptitle('US County Climate Data Coverage Analysis\n', fontsize=16, fontweight='bold')
    
    # Add color legend note
    fig.text(0.5, 0.02, 'Green = Climate data available | Red = Missing climate data', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = 'climate_outputs/missing_counties_map.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nMap saved to: {output_path}")
    
    # Also create a detailed state-level summary
    create_state_summary(counties_shp)
    
    plt.show()
    return counties_shp

def create_state_summary(counties_shp):
    """Create a detailed state-level summary of missing counties."""
    
    # Calculate statistics by state
    state_stats = counties_shp.groupby('STATEFP').agg({
        'has_data': ['sum', 'count']
    }).round(2)
    
    state_stats.columns = ['processed', 'total']
    state_stats['missing'] = state_stats['total'] - state_stats['processed']
    state_stats['coverage_pct'] = (state_stats['processed'] / state_stats['total'] * 100).round(1)
    
    # Sort by number missing
    state_stats = state_stats.sort_values('missing', ascending=False)
    
    # Save to CSV
    output_csv = 'climate_outputs/missing_counties_by_state.csv'
    state_stats.to_csv(output_csv)
    print(f"State summary saved to: {output_csv}")
    
    # Print top states with missing counties
    print("\nTop 10 States with Missing Counties:")
    print("-" * 50)
    print(f"{'State':<10} {'Total':<8} {'Missing':<8} {'Coverage':<10}")
    print("-" * 50)
    
    for state_code in state_stats.head(10).index:
        row = state_stats.loc[state_code]
        print(f"{state_code:<10} {int(row['total']):<8} {int(row['missing']):<8} {row['coverage_pct']:<10.1f}%")

if __name__ == "__main__":
    print("Creating Missing Counties Visualization...")
    print("=" * 50)
    
    try:
        counties_data = create_missing_counties_map()
        print("\n✅ Visualization complete!")
        
    except Exception as e:
        print(f"\n❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()