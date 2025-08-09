#!/usr/bin/env python
"""
Climate data visualization and reporting module.

Provides comprehensive visualization and reporting capabilities for 
processed climate statistics with interactive plots and summary reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ClimateVisualizer:
    """Main class for creating climate data visualizations and reports."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory for saving plots and reports
        """
        self.output_dir = output_dir or Path('./climate_reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        (self.output_dir / 'data_summaries').mkdir(exist_ok=True)
        
        # Color schemes for different regions
        self.region_colors = {
            'conus': '#2E86AB',
            'alaska': '#A23B72', 
            'hawaii': '#F18F01',
            'guam': '#C73E1D',
            'puerto_rico': '#1B998B'
        }
        
        self.loaded_data = {}
    
    def load_regional_data(self, data_paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
        """Load processed climate data for multiple regions.
        
        Args:
            data_paths: Dictionary mapping region names to CSV file paths
            
        Returns:
            Dictionary of loaded DataFrames by region
        """
        print("📊 Loading regional climate data...")
        
        for region, path in data_paths.items():
            if path.exists():
                df = pd.read_csv(path)
                df['region'] = region
                self.loaded_data[region] = df
                print(f"  ✅ {region.upper()}: {len(df):,} records ({df['year'].min()}-{df['year'].max()})")
            else:
                print(f"  ❌ {region.upper()}: File not found - {path}")
        
        return self.loaded_data
    
    def create_temporal_trends_plot(self, 
                                  variable: str = 'total_annual_precip_mm',
                                  regions: Optional[List[str]] = None,
                                  save_plot: bool = True) -> plt.Figure:
        """Create temporal trend plots showing climate variable changes over time.
        
        Args:
            variable: Climate variable to plot
            regions: List of regions to include (None for all)
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if not self.loaded_data:
            raise ValueError("No data loaded. Call load_regional_data() first.")
        
        regions = regions or list(self.loaded_data.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Climate Trends: {variable.replace("_", " ").title()} (2015-2100)', 
                    fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        # Plot 1: Regional annual averages over time
        ax1 = axes_flat[0]
        for region in regions:
            if region in self.loaded_data:
                df = self.loaded_data[region]
                annual_avg = df.groupby('year')[variable].mean()
                
                ax1.plot(annual_avg.index, annual_avg.values, 
                        linewidth=2.5, label=region.upper(), 
                        color=self.region_colors.get(region, 'gray'))
                
                # Add trend line
                z = np.polyfit(annual_avg.index, annual_avg.values, 1)
                p = np.poly1d(z)
                ax1.plot(annual_avg.index, p(annual_avg.index), 
                        linestyle='--', alpha=0.7, 
                        color=self.region_colors.get(region, 'gray'))
        
        ax1.set_title('Regional Annual Averages')
        ax1.set_xlabel('Year')
        ax1.set_ylabel(variable.replace('_', ' ').title())
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribution changes (early vs late period)
        ax2 = axes_flat[1]
        early_data = []
        late_data = []
        
        for region in regions:
            if region in self.loaded_data:
                df = self.loaded_data[region]
                early_period = df[df['year'] <= 2040]
                late_period = df[df['year'] >= 2070]
                
                early_data.extend(early_period[variable].values)
                late_data.extend(late_period[variable].values)
        
        ax2.hist(early_data, bins=50, alpha=0.7, label='2015-2040', density=True)
        ax2.hist(late_data, bins=50, alpha=0.7, label='2070-2100', density=True)
        ax2.set_title('Distribution Shift: Early vs Late Period')
        ax2.set_xlabel(variable.replace('_', ' ').title())
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Extreme events trends (if applicable)
        ax3 = axes_flat[2]
        if 'days_above_threshold' in self.loaded_data[list(self.loaded_data.keys())[0]].columns:
            for region in regions:
                if region in self.loaded_data:
                    df = self.loaded_data[region]
                    extreme_events = df.groupby('year')['days_above_threshold'].mean()
                    
                    ax3.plot(extreme_events.index, extreme_events.values,
                            linewidth=2.5, label=region.upper(),
                            color=self.region_colors.get(region, 'gray'))
            
            ax3.set_title('Extreme Events Trend (Days Above Threshold)')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('Average Days Above Threshold')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Extreme events data\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Extreme Events (Data Not Available)')
        
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Regional comparison boxplot
        ax4 = axes_flat[3]
        plot_data = []
        plot_regions = []
        
        for region in regions:
            if region in self.loaded_data:
                df = self.loaded_data[region]
                plot_data.append(df[variable].values)
                plot_regions.append(region.upper())
        
        if plot_data:
            box_plot = ax4.boxplot(plot_data, labels=plot_regions, patch_artist=True)
            
            # Color the boxes
            for patch, region in zip(box_plot['boxes'], regions):
                if region in self.region_colors:
                    patch.set_facecolor(self.region_colors[region])
                    patch.set_alpha(0.7)
        
        ax4.set_title('Regional Distribution Comparison')
        ax4.set_ylabel(variable.replace('_', ' ').title())
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / 'plots' / f'temporal_trends_{variable}_{timestamp}.png'
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📈 Saved temporal trends plot: {plot_path}")
        
        return fig
    
    def create_regional_comparison_plot(self, 
                                      metrics: List[str] = None,
                                      save_plot: bool = True) -> plt.Figure:
        """Create comprehensive regional comparison plots.
        
        Args:
            metrics: List of metrics to compare
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if not self.loaded_data:
            raise ValueError("No data loaded. Call load_regional_data() first.")
        
        # Default metrics to compare
        default_metrics = [
            'total_annual_precip_mm', 'mean_daily_precip_mm', 
            'days_above_threshold', 'max_daily_precip_mm'
        ]
        
        # Use available metrics from data
        available_columns = set()
        for df in self.loaded_data.values():
            available_columns.update(df.columns)
        
        metrics = metrics or [m for m in default_metrics if m in available_columns]
        
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        fig.suptitle('Regional Climate Comparison (2015-2100 Average)', 
                    fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Calculate regional averages
            regional_averages = {}
            regional_stds = {}
            
            for region, df in self.loaded_data.items():
                if metric in df.columns:
                    regional_averages[region] = df[metric].mean()
                    regional_stds[region] = df[metric].std()
            
            if regional_averages:
                regions = list(regional_averages.keys())
                values = list(regional_averages.values())
                errors = [regional_stds[r] for r in regions]
                colors = [self.region_colors.get(r, 'gray') for r in regions]
                
                bars = ax.bar(range(len(regions)), values, yerr=errors, 
                             capsize=5, color=colors, alpha=0.8)
                
                ax.set_xticks(range(len(regions)))
                ax.set_xticklabels([r.upper() for r in regions], rotation=45)
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # Hide empty subplots
        for i in range(len(metrics), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / 'plots' / f'regional_comparison_{timestamp}.png'
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved regional comparison plot: {plot_path}")
        
        return fig
    
    def create_climate_change_signals_plot(self, save_plot: bool = True) -> plt.Figure:
        """Create plots showing climate change signals over time.
        
        Args:
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if not self.loaded_data:
            raise ValueError("No data loaded. Call load_regional_data() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Climate Change Signals Analysis (SSP370 Scenario)', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Temperature/Precipitation trends with confidence intervals
        ax1 = axes[0, 0]
        for region, df in self.loaded_data.items():
            # Calculate decadal averages
            df['decade'] = (df['year'] // 10) * 10
            decadal_stats = df.groupby('decade')['total_annual_precip_mm'].agg(['mean', 'std'])
            
            decades = decadal_stats.index
            means = decadal_stats['mean']
            stds = decadal_stats['std']
            
            ax1.errorbar(decades, means, yerr=stds, 
                        label=region.upper(), linewidth=2.5, capsize=5,
                        color=self.region_colors.get(region, 'gray'))
        
        ax1.set_title('Decadal Precipitation Trends with Uncertainty')
        ax1.set_xlabel('Decade')
        ax1.set_ylabel('Annual Precipitation (mm)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Change relative to baseline (2015-2030)
        ax2 = axes[0, 1]
        for region, df in self.loaded_data.items():
            baseline = df[df['year'] <= 2030]['total_annual_precip_mm'].mean()
            annual_means = df.groupby('year')['total_annual_precip_mm'].mean()
            relative_change = ((annual_means - baseline) / baseline * 100)
            
            ax2.plot(relative_change.index, relative_change.values,
                    linewidth=2.5, label=region.upper(),
                    color=self.region_colors.get(region, 'gray'))
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Change Relative to 2015-2030 Baseline (%)')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Percent Change (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Extreme events frequency
        ax3 = axes[1, 0]
        if 'days_above_threshold' in df.columns:
            for region, df in self.loaded_data.items():
                # Calculate rolling 10-year average of extreme events
                annual_extremes = df.groupby('year')['days_above_threshold'].mean()
                rolling_avg = annual_extremes.rolling(window=10, center=True).mean()
                
                ax3.plot(rolling_avg.index, rolling_avg.values,
                        linewidth=2.5, label=region.upper(),
                        color=self.region_colors.get(region, 'gray'))
            
            ax3.set_title('Extreme Events Frequency (10-year rolling average)')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('Days Above Threshold')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Extreme events data\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Extreme Events (Data Not Available)')
        
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Variability changes (coefficient of variation)
        ax4 = axes[1, 1]
        for region, df in self.loaded_data.items():
            # Calculate 10-year rolling coefficient of variation
            annual_means = df.groupby('year')['total_annual_precip_mm'].mean()
            rolling_cv = (annual_means.rolling(window=10).std() / 
                         annual_means.rolling(window=10).mean() * 100)
            
            ax4.plot(rolling_cv.index, rolling_cv.values,
                    linewidth=2.5, label=region.upper(),
                    color=self.region_colors.get(region, 'gray'))
        
        ax4.set_title('Climate Variability Changes (Coefficient of Variation)')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Coefficient of Variation (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / 'plots' / f'climate_change_signals_{timestamp}.png'
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"🌡️ Saved climate change signals plot: {plot_path}")
        
        return fig
    
    def generate_summary_report(self, save_report: bool = True) -> Dict:
        """Generate comprehensive summary report of climate data analysis.
        
        Args:
            save_report: Whether to save the report
            
        Returns:
            Dictionary containing summary statistics
        """
        if not self.loaded_data:
            raise ValueError("No data loaded. Call load_regional_data() first.")
        
        print("📋 Generating comprehensive climate summary report...")
        
        report = {
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'regions_analyzed': list(self.loaded_data.keys()),
                'total_records': sum(len(df) for df in self.loaded_data.values()),
                'analysis_period': f"{min(df['year'].min() for df in self.loaded_data.values())}-{max(df['year'].max() for df in self.loaded_data.values())}"
            },
            'regional_summaries': {},
            'comparative_analysis': {},
            'climate_trends': {},
            'extreme_events': {}
        }
        
        # Regional summaries
        for region, df in self.loaded_data.items():
            summary = {
                'data_overview': {
                    'total_records': len(df),
                    'counties': df['county_id'].nunique() if 'county_id' in df.columns else 'N/A',
                    'years_covered': f"{df['year'].min()}-{df['year'].max()}",
                    'data_completeness': f"{(1 - df.isnull().sum().sum() / df.size) * 100:.1f}%"
                },
                'precipitation_statistics': {
                    'mean_annual_precip_mm': float(df['total_annual_precip_mm'].mean()),
                    'median_annual_precip_mm': float(df['total_annual_precip_mm'].median()),
                    'std_annual_precip_mm': float(df['total_annual_precip_mm'].std()),
                    'min_annual_precip_mm': float(df['total_annual_precip_mm'].min()),
                    'max_annual_precip_mm': float(df['total_annual_precip_mm'].max()),
                    'mean_daily_precip_mm': float(df['mean_daily_precip_mm'].mean()),
                    'extreme_precip_days_avg': float(df['days_above_threshold'].mean()) if 'days_above_threshold' in df.columns else None
                }
            }
            
            # Climate trends analysis
            annual_means = df.groupby('year')['total_annual_precip_mm'].mean()
            trend_slope = np.polyfit(annual_means.index, annual_means.values, 1)[0]
            
            summary['climate_trends'] = {
                'precipitation_trend_mm_per_year': float(trend_slope),
                'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
                'trend_magnitude': abs(float(trend_slope * 85)),  # Total change over 85 years
                'baseline_vs_endperiod': {
                    'baseline_2015_2030': float(df[df['year'] <= 2030]['total_annual_precip_mm'].mean()),
                    'end_period_2085_2100': float(df[df['year'] >= 2085]['total_annual_precip_mm'].mean()),
                    'percent_change': float(((df[df['year'] >= 2085]['total_annual_precip_mm'].mean() - 
                                            df[df['year'] <= 2030]['total_annual_precip_mm'].mean()) / 
                                           df[df['year'] <= 2030]['total_annual_precip_mm'].mean()) * 100)
                }
            }
            
            report['regional_summaries'][region] = summary
        
        # Comparative analysis
        all_data = pd.concat(self.loaded_data.values(), ignore_index=True)
        
        regional_means = {}
        for region, df in self.loaded_data.items():
            regional_means[region] = df['total_annual_precip_mm'].mean()
        
        report['comparative_analysis'] = {
            'wettest_region': max(regional_means, key=regional_means.get),
            'driest_region': min(regional_means, key=regional_means.get),
            'regional_precipitation_ranking': sorted(regional_means.items(), key=lambda x: x[1], reverse=True),
            'coefficient_of_variation_by_region': {
                region: float(df['total_annual_precip_mm'].std() / df['total_annual_precip_mm'].mean())
                for region, df in self.loaded_data.items()
            }
        }
        
        # Overall trends
        overall_annual = all_data.groupby(['year', 'region'])['total_annual_precip_mm'].mean().groupby('year').mean()
        overall_trend = np.polyfit(overall_annual.index, overall_annual.values, 1)[0]
        
        report['climate_trends']['overall'] = {
            'us_average_trend_mm_per_year': float(overall_trend),
            'projected_change_2015_2100': float(overall_trend * 85),
            'trend_significance': 'significant' if abs(overall_trend) > 1 else 'minimal'
        }
        
        # Extreme events analysis
        if 'days_above_threshold' in all_data.columns:
            extreme_trends = {}
            for region, df in self.loaded_data.items():
                annual_extremes = df.groupby('year')['days_above_threshold'].mean()
                extreme_trend = np.polyfit(annual_extremes.index, annual_extremes.values, 1)[0]
                extreme_trends[region] = {
                    'trend_days_per_year': float(extreme_trend),
                    'baseline_extreme_days': float(df[df['year'] <= 2030]['days_above_threshold'].mean()),
                    'future_extreme_days': float(df[df['year'] >= 2085]['days_above_threshold'].mean())
                }
            
            report['extreme_events'] = extreme_trends
        
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / 'reports' / f'climate_summary_report_{timestamp}.json'
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"📄 Saved comprehensive report: {report_path}")
            
            # Also create a human-readable summary
            self._create_readable_summary(report, timestamp)
        
        return report
    
    def _create_readable_summary(self, report: Dict, timestamp: str):
        """Create a human-readable summary report."""
        readable_path = self.output_dir / 'reports' / f'climate_summary_readable_{timestamp}.txt'
        
        with open(readable_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CLIMATE DATA ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {report['generation_info']['timestamp']}\n")
            f.write(f"Analysis Period: {report['generation_info']['analysis_period']}\n")
            f.write(f"Regions Analyzed: {', '.join(report['generation_info']['regions_analyzed'])}\n")
            f.write(f"Total Records: {report['generation_info']['total_records']:,}\n\n")
            
            f.write("REGIONAL SUMMARIES\n")
            f.write("-" * 40 + "\n\n")
            
            for region, summary in report['regional_summaries'].items():
                f.write(f"{region.upper()}\n")
                f.write(f"  Records: {summary['data_overview']['total_records']:,}\n")
                f.write(f"  Counties: {summary['data_overview']['counties']}\n")
                f.write(f"  Mean Annual Precipitation: {summary['precipitation_statistics']['mean_annual_precip_mm']:.1f} mm\n")
                f.write(f"  Precipitation Trend: {summary['climate_trends']['precipitation_trend_mm_per_year']:.2f} mm/year ({summary['climate_trends']['trend_direction']})\n")
                f.write(f"  Total Change (2015-2100): {summary['climate_trends']['baseline_vs_endperiod']['percent_change']:.1f}%\n\n")
            
            f.write("COMPARATIVE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Wettest Region: {report['comparative_analysis']['wettest_region'].upper()}\n")
            f.write(f"Driest Region: {report['comparative_analysis']['driest_region'].upper()}\n\n")
            
            f.write("Regional Precipitation Ranking:\n")
            for i, (region, precip) in enumerate(report['comparative_analysis']['regional_precipitation_ranking'], 1):
                f.write(f"  {i}. {region.upper()}: {precip:.1f} mm/year\n")
            
            f.write(f"\nOverall US Trend: {report['climate_trends']['overall']['us_average_trend_mm_per_year']:.2f} mm/year\n")
            f.write(f"Projected Change (2015-2100): {report['climate_trends']['overall']['projected_change_2015_2100']:.1f} mm\n")
        
        print(f"📖 Saved readable summary: {readable_path}")


def discover_processed_data(base_dir: Path = None) -> Dict[str, Path]:
    """Automatically discover processed climate data files.
    
    Args:
        base_dir: Base directory to search (defaults to climate_outputs)
        
    Returns:
        Dictionary mapping region names to data file paths
    """
    base_dir = base_dir or Path('./climate_outputs')
    data_paths = {}
    
    if not base_dir.exists():
        print(f"❌ Base directory not found: {base_dir}")
        return data_paths
    
    # Search for precipitation statistics files
    stats_dir = base_dir / 'stats' / 'pr'
    
    if stats_dir.exists():
        for region_dir in stats_dir.iterdir():
            if region_dir.is_dir():
                region_name = region_dir.name
                
                # Look for SSP370 scenario files (most recent processing)
                ssp370_dir = region_dir / 'ssp370'
                if ssp370_dir.exists():
                    # Find the most recent or comprehensive stats file
                    csv_files = [f for f in ssp370_dir.glob('*.csv') if not f.name.startswith('._')]
                    if csv_files:
                        # Prefer threshold files, then others
                        threshold_files = [f for f in csv_files if 'threshold' in f.name]
                        if threshold_files:
                            data_paths[region_name] = threshold_files[0]
                        else:
                            data_paths[region_name] = csv_files[0]
    
    return data_paths


def main():
    """Main function for running climate visualization and reporting."""
    print("🌡️ Climate Data Visualization and Reporting")
    print("=" * 50)
    
    # Discover processed data
    data_paths = discover_processed_data()
    
    if not data_paths:
        print("❌ No processed climate data found.")
        print("Please run the climate processing pipeline first.")
        return
    
    print(f"📁 Found data for {len(data_paths)} regions:")
    for region, path in data_paths.items():
        print(f"  • {region.upper()}: {path}")
    
    # Create visualizer
    visualizer = ClimateVisualizer()
    
    # Load data
    visualizer.load_regional_data(data_paths)
    
    # Generate visualizations
    print("\n🎨 Creating visualizations...")
    
    try:
        # Temporal trends
        visualizer.create_temporal_trends_plot(variable='total_annual_precip_mm')
        
        # Regional comparison
        visualizer.create_regional_comparison_plot()
        
        # Climate change signals
        visualizer.create_climate_change_signals_plot()
        
        # Generate comprehensive report
        print("\n📊 Generating comprehensive report...")
        report = visualizer.generate_summary_report()
        
        print(f"\n✅ Analysis complete!")
        print(f"📁 Reports saved to: {visualizer.output_dir}")
        print(f"📈 Generated {len(list((visualizer.output_dir / 'plots').glob('*.png')))} plots")
        print(f"📄 Generated {len(list((visualizer.output_dir / 'reports').glob('*')))} reports")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()