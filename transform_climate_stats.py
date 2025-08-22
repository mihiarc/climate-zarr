#!/usr/bin/env python3
"""
Climate Statistics Transformation Script

Transforms county-level climate statistics from individual variable files 
into a standardized output format matching climate_output_format.yaml.

This script:
1. Reads all CSV files from climate_outputs/stats/ directories
2. Merges data by county_id and year across variables  
3. Maps existing columns to target format
4. Handles missing data appropriately
5. Formats county names as "COUNTY, STATE"

Author: Climate Zarr Toolkit
Date: 2025-08-22
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import yaml

# Configure warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Setup rich console for pretty output
console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('climate_transform.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ClimateDataTransformer:
    """
    Transforms county-level climate statistics into standardized format.
    
    Handles merging multiple climate variables (precipitation, temperature) 
    across different regions (CONUS, Alaska, Hawaii, Puerto Rico, Guam)
    into a unified output format.
    """
    
    def __init__(self, 
                 stats_dir: str = "climate_outputs/stats",
                 output_dir: str = "climate_outputs/transformed",
                 format_spec: str = "climate_output_format.yaml"):
        """
        Initialize the transformer.
        
        Args:
            stats_dir: Path to directory containing stats CSV files
            output_dir: Path to output directory for transformed files
            format_spec: Path to YAML file with output format specification
        """
        self.stats_dir = Path(stats_dir)
        self.output_dir = Path(output_dir)
        self.format_spec_path = Path(format_spec)
        
        # Load output format specification
        self.target_format = self._load_format_spec()
        
        # Climate variable mappings
        self.variable_mappings = {
            'pr': {
                'source_pattern': '*pr*stats*.csv',
                'key_columns': ['days_above_threshold', 'total_annual_precip_mm'],
                'target_mappings': {
                    'days_above_threshold': 'daysabove1in',
                    'total_annual_precip_mm': 'annual_total_precip'
                }
            },
            'tas': {
                'source_pattern': '*tas*stats*.csv',
                'key_columns': ['mean_annual_temp_c'],
                'target_mappings': {
                    'mean_annual_temp_c': 'annual_mean_temp'
                }
            },
            'tasmax': {
                'source_pattern': '*tasmax*stats*.csv', 
                'key_columns': ['mean_annual_tasmax_c', 'days_above_threshold_c'],
                'target_mappings': {
                    'mean_annual_tasmax_c': 'tmaxavg',
                    'days_above_threshold_c': 'daysabove90F'  # Will need threshold adjustment
                }
            },
            'tasmin': {
                'source_pattern': '*tasmin*stats*.csv',
                'key_columns': ['mean_annual_tasmin_c'],
                'target_mappings': {
                    # No direct mapping to target format, but useful for validation
                }
            }
        }
        
        # Regions to process
        self.regions = ['conus', 'alaska', 'hawaii', 'puerto_rico', 'guam']
        
        # Results storage
        self.processing_results = {}
        self.transformation_metadata = {}
        
    def _load_format_spec(self) -> Dict:
        """Load the output format specification from YAML."""
        try:
            with open(self.format_spec_path, 'r') as f:
                spec = yaml.safe_load(f)
            logger.info(f"Loaded format specification from {self.format_spec_path}")
            return spec
        except FileNotFoundError:
            logger.error(f"Format specification file not found: {self.format_spec_path}")
            # Return default format if file not found
            return {
                'columns': [
                    {'name': 'cid2', 'type': 'integer'},
                    {'name': 'year', 'type': 'integer'},
                    {'name': 'name', 'type': 'string'},
                    {'name': 'daysabove1in', 'type': 'integer'},
                    {'name': 'daysabove90F', 'type': 'integer'},
                    {'name': 'tmaxavg', 'type': 'numeric'},
                    {'name': 'annual_mean_temp', 'type': 'numeric'},
                    {'name': 'annual_total_precip', 'type': 'numeric'}
                ]
            }
    
    def discover_files(self) -> Dict[str, List[Path]]:
        """
        Discover all available climate statistics files.
        
        Returns:
            Dictionary mapping variable names to lists of file paths
        """
        discovered_files = {var: [] for var in self.variable_mappings.keys()}
        
        console.print("\n[bold blue]Discovering climate statistics files...[/bold blue]")
        
        for variable in self.variable_mappings.keys():
            pattern = self.variable_mappings[variable]['source_pattern']
            
            # Search in all subdirectories
            files = list(self.stats_dir.rglob(pattern))
            discovered_files[variable] = files
            
            if files:
                console.print(f"  ✓ Found {len(files)} {variable.upper()} files")
                for file in files:
                    logger.info(f"    {file}")
            else:
                console.print(f"  ⚠ No {variable.upper()} files found (pattern: {pattern})")
        
        return discovered_files
    
    def load_climate_data(self, files: Dict[str, List[Path]]) -> Dict[str, pd.DataFrame]:
        """
        Load all climate data files into DataFrames.
        
        Args:
            files: Dictionary mapping variable names to file paths
            
        Returns:
            Dictionary mapping variable names to concatenated DataFrames
        """
        climate_data = {}
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            # Count total files to process
            total_files = sum(len(file_list) for file_list in files.values())
            load_task = progress.add_task("Loading climate data files", total=total_files)
            
            for variable, file_list in files.items():
                if not file_list:
                    console.print(f"  [yellow]Skipping {variable.upper()} - no files found[/yellow]")
                    continue
                
                dfs = []
                for file_path in file_list:
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Add metadata columns
                        df['source_file'] = file_path.name
                        df['variable'] = variable
                        df['region'] = self._extract_region_from_path(file_path)
                        
                        dfs.append(df)
                        progress.advance(load_task)
                        
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
                        progress.advance(load_task)
                        continue
                
                if dfs:
                    # Concatenate all files for this variable
                    climate_data[variable] = pd.concat(dfs, ignore_index=True)
                    console.print(f"  ✓ Loaded {len(climate_data[variable])} records for {variable.upper()}")
                    
                    # Log basic statistics
                    logger.info(f"{variable.upper()} data shape: {climate_data[variable].shape}")
                    logger.info(f"  Years: {climate_data[variable]['year'].min()} - {climate_data[variable]['year'].max()}")
                    logger.info(f"  Unique counties: {climate_data[variable]['county_id'].nunique()}")
        
        return climate_data
    
    def _extract_region_from_path(self, file_path: Path) -> str:
        """Extract region name from file path."""
        path_parts = file_path.parts
        for region in self.regions:
            if region in path_parts:
                return region
        return 'unknown'
    
    def _calculate_daysabove90F(self, tasmax_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate days above 90°F (32.2°C) from tasmax data.
        
        The current tasmax files have days_above_threshold_c for 35°C.
        We need to calculate days above 32.2°C (90°F).
        
        Since we don't have access to the raw daily data, we'll need to 
        estimate or use the existing threshold data as a proxy.
        """
        # For now, we'll use the existing days_above_35c column as it's the closest
        # In a production system, you'd want to reprocess with the correct threshold
        
        if 'days_above_35c' in tasmax_data.columns:
            # Use the existing 35°C threshold as a conservative estimate
            tasmax_data['daysabove90F'] = tasmax_data['days_above_35c']
            logger.warning("Using days_above_35c as proxy for days_above_90F (32.2°C). "
                         "For precise results, reprocess with 32.2°C threshold.")
        elif 'days_above_threshold_c' in tasmax_data.columns:
            # Check if the threshold is 32.2°C
            if 'threshold_temp_c' in tasmax_data.columns:
                # Check if any records have 32.2°C threshold
                correct_threshold = tasmax_data['threshold_temp_c'] == 32.2
                if correct_threshold.any():
                    tasmax_data['daysabove90F'] = tasmax_data['days_above_threshold_c']
                else:
                    # Use the existing threshold as proxy
                    tasmax_data['daysabove90F'] = tasmax_data['days_above_threshold_c']
                    logger.warning(f"No 32.2°C threshold found. Using existing threshold "
                                 f"({tasmax_data['threshold_temp_c'].iloc[0]}°C) as proxy.")
            else:
                tasmax_data['daysabove90F'] = tasmax_data['days_above_threshold_c']
        else:
            # No threshold data available, set to NaN
            tasmax_data['daysabove90F'] = np.nan
            logger.warning("No threshold temperature data found in tasmax files. "
                         "Setting daysabove90F to NaN.")
        
        return tasmax_data
    
    def transform_data(self, climate_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Transform climate data to target format.
        
        Args:
            climate_data: Dictionary of DataFrames by variable
            
        Returns:
            Merged and transformed DataFrame
        """
        console.print("\n[bold blue]Transforming data to target format...[/bold blue]")
        
        # Process each variable separately to create clean datasets
        processed_variables = {}
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            transform_task = progress.add_task("Processing variables", total=len(climate_data))
            
            for variable, df in climate_data.items():
                logger.info(f"Processing {variable.upper()} data...")
                
                # Clean and prepare the data
                var_data = df.copy()
                
                # Create a unique key for each record (including scenario)
                scenario_col = var_data['scenario'] if 'scenario' in var_data.columns else 'unknown'
                var_data['merge_key'] = (
                    var_data['county_id'].astype(str) + '_' + 
                    var_data['year'].astype(str) + '_' + 
                    var_data['region'].astype(str) + '_' +
                    scenario_col.astype(str)
                )
                
                # Remove duplicates by keeping the first occurrence
                before_dedup = len(var_data)
                var_data = var_data.drop_duplicates(subset=['merge_key'], keep='first')
                after_dedup = len(var_data)
                
                if before_dedup != after_dedup:
                    logger.warning(f"  Removed {before_dedup - after_dedup} duplicate records from {variable.upper()}")
                
                # Special handling for tasmax to calculate daysabove90F
                if variable == 'tasmax':
                    var_data = self._calculate_daysabove90F(var_data)
                
                # Create the target columns for this variable (including scenario)
                base_columns = ['merge_key', 'county_id', 'year', 'county_name', 'state', 'region']
                if 'scenario' in var_data.columns:
                    base_columns.insert(3, 'scenario')  # Insert scenario after year
                target_data = var_data[base_columns].copy()
                
                # Add variable-specific columns based on mappings
                mappings = self.variable_mappings[variable]['target_mappings']
                for source_col, target_col in mappings.items():
                    if source_col in var_data.columns:
                        target_data[target_col] = var_data[source_col]
                        logger.debug(f"    Mapped {source_col} -> {target_col}")
                
                # For tasmax, also add the calculated daysabove90F
                if variable == 'tasmax' and 'daysabove90F' in var_data.columns:
                    target_data['daysabove90F'] = var_data['daysabove90F']
                
                processed_variables[variable] = target_data
                logger.info(f"  Processed {len(target_data)} records for {variable.upper()}")
                
                progress.advance(transform_task)
        
        if not processed_variables:
            raise ValueError("No data available for transformation")
        
        # Start merging from the largest dataset (typically TAS)
        variable_names = list(processed_variables.keys())
        variable_sizes = {var: len(df) for var, df in processed_variables.items()}
        primary_variable = max(variable_sizes, key=variable_sizes.get)
        
        logger.info(f"Using {primary_variable.upper()} as primary dataset ({variable_sizes[primary_variable]} records)")
        merged_data = processed_variables[primary_variable].copy()
        
        # Merge other variables
        for variable in variable_names:
            if variable == primary_variable:
                continue
                
            var_data = processed_variables[variable]
            
            # Merge on the unique key (drop duplicated columns but keep scenario from primary)
            columns_to_drop = ['county_id', 'year', 'county_name', 'state', 'region']
            if 'scenario' in var_data.columns and 'scenario' in merged_data.columns:
                columns_to_drop.append('scenario')
            
            before_merge = len(merged_data)
            merged_data = merged_data.merge(
                var_data.drop(columns=columns_to_drop, errors='ignore'),
                on='merge_key',
                how='left',
                suffixes=('', f'_{variable}')
            )
            
            logger.info(f"  Merged {variable.upper()}: {before_merge} -> {len(merged_data)} records")
        
        # Remove the merge key but keep region for later use
        merged_data = merged_data.drop(columns=['merge_key'])
        
        # Apply final transformations
        merged_data = self._apply_final_transformations(merged_data)
        
        console.print(f"  ✓ Transformation complete: {len(merged_data)} total records")
        return merged_data
    
    def _apply_final_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final column transformations to match target format."""
        
        # Create standardized columns
        result_df = df.copy()
        
        # 1. Create cid2 (FIPS county code)
        result_df['cid2'] = pd.to_numeric(result_df['county_id'], errors='coerce').astype('Int64')
        
        # 2. Format county name as "COUNTY, STATE"
        if 'county_name' in result_df.columns and 'state' in result_df.columns:
            # Handle missing state information
            state_col = result_df['state'].fillna('')
            county_col = result_df['county_name'].fillna('')
            
            # Create formatted name
            result_df['name'] = county_col + ', ' + state_col
            
            # Clean up names where state is missing
            result_df['name'] = result_df['name'].str.replace(', $', '', regex=True)
        else:
            logger.warning("County name or state column missing. Using county_id as name.")
            result_df['name'] = result_df['county_id'].astype(str)
        
        # 3. Ensure required columns exist with appropriate defaults
        target_columns = [col['name'] for col in self.target_format['columns']]
        
        for col_spec in self.target_format['columns']:
            col_name = col_spec['name']
            col_type = col_spec['type']
            
            if col_name not in result_df.columns:
                # Create missing columns with appropriate defaults
                if col_type == 'integer':
                    result_df[col_name] = pd.NA
                elif col_type == 'numeric':
                    result_df[col_name] = np.nan
                else:  # string
                    result_df[col_name] = ''
                
                logger.warning(f"Column '{col_name}' not found in data. Created with default values.")
        
        # 4. Select and order columns according to target format (keep region for later)
        final_columns = [col['name'] for col in self.target_format['columns']]
        available_columns = [col for col in final_columns if col in result_df.columns]
        
        # Keep region column if it exists for potential separation later
        if 'region' in result_df.columns and 'region' not in available_columns:
            available_columns.append('region')
        
        result_df = result_df[available_columns]
        
        # 5. Apply data type conversions
        for col_spec in self.target_format['columns']:
            col_name = col_spec['name']
            col_type = col_spec['type']
            
            if col_name in result_df.columns:
                try:
                    if col_type == 'integer':
                        result_df[col_name] = pd.to_numeric(result_df[col_name], errors='coerce').astype('Int64')
                    elif col_type == 'numeric':
                        result_df[col_name] = pd.to_numeric(result_df[col_name], errors='coerce')
                    else:  # string
                        result_df[col_name] = result_df[col_name].astype(str)
                except Exception as e:
                    logger.warning(f"Error converting column '{col_name}' to {col_type}: {e}")
        
        return result_df
    
    def validate_output(self, df: pd.DataFrame) -> Dict[str, Union[int, float, List]]:
        """
        Validate the transformed output and generate statistics.
        
        Args:
            df: Transformed DataFrame
            
        Returns:
            Dictionary with validation results and statistics
        """
        console.print("\n[bold blue]Validating output data...[/bold blue]")
        
        validation_results = {
            'total_records': len(df),
            'unique_counties': df['cid2'].nunique(),
            'year_range': [int(df['year'].min()), int(df['year'].max())] if len(df) > 0 else [None, None],
            'scenarios': df['scenario'].unique().tolist() if 'scenario' in df.columns else [],
            'missing_data': {},
            'data_quality_issues': []
        }
        
        # Check for missing data in each column
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            validation_results['missing_data'][col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }
            
            if missing_pct > 50:
                validation_results['data_quality_issues'].append(
                    f"Column '{col}' has {missing_pct:.1f}% missing data"
                )
        
        # Check for duplicate records (including scenario if present)
        dup_cols = ['cid2', 'year']
        if 'scenario' in df.columns:
            dup_cols.append('scenario')
        
        duplicate_mask = df.duplicated(subset=dup_cols)
        if duplicate_mask.any():
            dup_count = duplicate_mask.sum()
            validation_results['data_quality_issues'].append(
                f"{dup_count} duplicate county-year-scenario combinations found"
            )
        
        # Check FIPS code validity (should be 5-digit integers)
        invalid_fips = df['cid2'].isna() | (df['cid2'] < 1000) | (df['cid2'] > 99999)
        if invalid_fips.any():
            invalid_count = invalid_fips.sum()
            validation_results['data_quality_issues'].append(
                f"{invalid_count} invalid FIPS codes found"
            )
        
        # Display validation summary
        table = Table(title="Data Validation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Records", f"{validation_results['total_records']:,}")
        table.add_row("Unique Counties", f"{validation_results['unique_counties']:,}")
        table.add_row("Year Range", f"{validation_results['year_range'][0]} - {validation_results['year_range'][1]}")
        table.add_row("Data Quality Issues", f"{len(validation_results['data_quality_issues'])}")
        
        console.print(table)
        
        # Display missing data summary
        if validation_results['missing_data']:
            missing_table = Table(title="Missing Data by Column")
            missing_table.add_column("Column", style="cyan")
            missing_table.add_column("Missing Count", style="red")
            missing_table.add_column("Missing %", style="red")
            
            for col, stats in validation_results['missing_data'].items():
                missing_table.add_row(
                    col,
                    f"{stats['count']:,}",
                    f"{stats['percentage']:.1f}%"
                )
            
            console.print(missing_table)
        
        # Display data quality issues
        if validation_results['data_quality_issues']:
            console.print("\n[bold red]Data Quality Issues:[/bold red]")
            for issue in validation_results['data_quality_issues']:
                console.print(f"  • {issue}")
        else:
            console.print("\n[bold green]✓ No major data quality issues found[/bold green]")
        
        return validation_results
    
    def save_results(self, 
                    df: pd.DataFrame, 
                    validation_results: Dict,
                    output_filename: str = "transformed_climate_stats.csv") -> Path:
        """
        Save transformed data and metadata.
        
        Args:
            df: Transformed DataFrame
            validation_results: Validation results dictionary
            output_filename: Name for the output CSV file
            
        Returns:
            Path to the saved file
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main CSV file
        output_path = self.output_dir / output_filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved transformed data to {output_path}")
        
        # Save metadata
        metadata = {
            'transformation_date': pd.Timestamp.now().isoformat(),
            'source_directory': str(self.stats_dir),
            'output_file': str(output_path),
            'format_specification': str(self.format_spec_path),
            'validation_results': validation_results,
            'column_mappings': self.variable_mappings,
            'regions_processed': self.regions
        }
        
        metadata_path = self.output_dir / f"{output_filename.replace('.csv', '_metadata.json')}"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Save validation report
        report_path = self.output_dir / f"{output_filename.replace('.csv', '_validation_report.txt')}"
        with open(report_path, 'w') as f:
            f.write("Climate Data Transformation Validation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Transformation Date: {metadata['transformation_date']}\n")
            f.write(f"Source Directory: {metadata['source_directory']}\n")
            f.write(f"Output File: {metadata['output_file']}\n\n")
            
            f.write("Summary Statistics:\n")
            f.write(f"  Total Records: {validation_results['total_records']:,}\n")
            f.write(f"  Unique Counties: {validation_results['unique_counties']:,}\n")
            f.write(f"  Year Range: {validation_results['year_range'][0]} - {validation_results['year_range'][1]}\n\n")
            
            f.write("Missing Data by Column:\n")
            for col, stats in validation_results['missing_data'].items():
                f.write(f"  {col}: {stats['count']:,} ({stats['percentage']:.1f}%)\n")
            
            if validation_results['data_quality_issues']:
                f.write("\nData Quality Issues:\n")
                for issue in validation_results['data_quality_issues']:
                    f.write(f"  • {issue}\n")
            else:
                f.write("\nNo major data quality issues found.\n")
        
        logger.info(f"Saved validation report to {report_path}")
        
        return output_path
    
    def run_transformation(self, 
                          output_filename: str = "transformed_climate_stats.csv",
                          separate_by_region: bool = False) -> Union[Path, List[Path]]:
        """
        Run the complete transformation pipeline.
        
        Args:
            output_filename: Name for the output file(s)
            separate_by_region: If True, create separate files per region
            
        Returns:
            Path(s) to output file(s)
        """
        console.print(Panel.fit(
            "[bold blue]Climate Data Transformation Pipeline[/bold blue]\n"
            f"Source: {self.stats_dir}\n"
            f"Output: {self.output_dir}\n"
            f"Format: {self.format_spec_path}",
            title="🌡️ Climate Zarr Toolkit"
        ))
        
        try:
            # Step 1: Discover files
            files = self.discover_files()
            total_files = sum(len(file_list) for file_list in files.values())
            
            if total_files == 0:
                console.print("[bold red]❌ No climate statistics files found![/bold red]")
                console.print(f"Check that files exist in: {self.stats_dir}")
                return None
            
            console.print(f"\n[bold green]✓ Found {total_files} climate statistics files[/bold green]")
            
            # Step 2: Load data
            climate_data = self.load_climate_data(files)
            
            if not climate_data:
                console.print("[bold red]❌ No data could be loaded![/bold red]")
                return None
            
            # Step 3: Transform data
            transformed_data = self.transform_data(climate_data)
            
            # Step 4: Validate results
            validation_results = self.validate_output(transformed_data)
            
            # Step 5: Save results
            if separate_by_region:
                output_paths = []
                
                # Check if region column exists
                if 'region' not in transformed_data.columns:
                    console.print("[bold red]❌ Region column not found. Cannot separate by region.[/bold red]")
                    console.print("Saving as single file instead...")
                    final_data = transformed_data.copy()
                    output_path = self.save_results(final_data, validation_results, output_filename)
                    return output_path
                
                for region in transformed_data['region'].unique():
                    if pd.isna(region):
                        continue
                    
                    region_data = transformed_data[transformed_data['region'] == region].copy()
                    # Remove region column from output
                    region_data = region_data.drop(columns=['region'], errors='ignore')
                    
                    region_filename = f"{region}_{output_filename}"
                    region_validation = self.validate_output(region_data)
                    
                    output_path = self.save_results(
                        region_data, 
                        region_validation, 
                        region_filename
                    )
                    output_paths.append(output_path)
                    
                    console.print(f"  ✓ Saved {region} data: {len(region_data)} records")
                
                return output_paths
            else:
                # Remove region column from final output if not needed
                final_data = transformed_data.drop(columns=['region'], errors='ignore')
                output_path = self.save_results(final_data, validation_results, output_filename)
                
                console.print(f"\n[bold green]✅ Transformation complete![/bold green]")
                console.print(f"Output saved to: {output_path}")
                
                return output_path
                
        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            console.print(f"[bold red]❌ Transformation failed: {e}[/bold red]")
            raise


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Transform climate statistics to standardized format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transformation
  python transform_climate_stats.py
  
  # Custom directories
  python transform_climate_stats.py --stats-dir custom_stats --output-dir custom_output
  
  # Separate files by region
  python transform_climate_stats.py --separate-by-region
  
  # Custom output filename
  python transform_climate_stats.py --output climate_data_2025.csv
        """
    )
    
    parser.add_argument(
        '--stats-dir',
        default='climate_outputs/stats',
        help='Directory containing climate statistics CSV files (default: climate_outputs/stats)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='climate_outputs/transformed',
        help='Output directory for transformed files (default: climate_outputs/transformed)'
    )
    
    parser.add_argument(
        '--format-spec',
        default='climate_output_format.yaml',
        help='YAML file with output format specification (default: climate_output_format.yaml)'
    )
    
    parser.add_argument(
        '--output',
        default='transformed_climate_stats.csv',
        help='Output filename (default: transformed_climate_stats.csv)'
    )
    
    parser.add_argument(
        '--separate-by-region',
        action='store_true',
        help='Create separate output files for each region'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize transformer
    transformer = ClimateDataTransformer(
        stats_dir=args.stats_dir,
        output_dir=args.output_dir,
        format_spec=args.format_spec
    )
    
    # Run transformation
    try:
        output_paths = transformer.run_transformation(
            output_filename=args.output,
            separate_by_region=args.separate_by_region
        )
        
        if output_paths:
            console.print("\n[bold green]🎉 Transformation completed successfully![/bold green]")
            if isinstance(output_paths, list):
                console.print(f"Created {len(output_paths)} regional files:")
                for path in output_paths:
                    console.print(f"  • {path}")
            else:
                console.print(f"Created output file: {output_paths}")
        else:
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()