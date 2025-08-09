#!/usr/bin/env python
"""Utilities for standardized output file and directory management."""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import json
import logging
from datetime import datetime

from climate_zarr.climate_config import get_config, ClimateConfig

logger = logging.getLogger(__name__)


class OutputManager:
    """Manages standardized output files and directories."""
    
    def __init__(self, config: Optional[ClimateConfig] = None):
        """Initialize output manager with configuration."""
        self.config = config or get_config()
        self.config.setup_directories()
    
    def get_output_path(self,
                       variable: str,
                       region: str,
                       scenario: str = "historical",
                       output_type: str = "stats",
                       file_extension: str = "csv",
                       custom_suffix: Optional[str] = None,
                       threshold: Optional[float] = None) -> Path:
        """Get standardized output path for a file."""
        return self.config.output.get_full_output_path(
            variable=variable,
            region=region,
            scenario=scenario,
            output_type=output_type,
            file_extension=file_extension,
            custom_suffix=custom_suffix,
            threshold=threshold
        )
    
    def create_output_directory(self, output_path: Path) -> Path:
        """Create output directory and return the path."""
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        return output_dir
    
    def save_with_metadata(self,
                          data: Any,
                          output_path: Path,
                          metadata: Optional[Dict] = None,
                          save_method: str = "auto") -> Path:
        """Save data with optional metadata file."""
        # Create directory
        self.create_output_directory(output_path)
        
        # Save main data file
        if save_method == "auto":
            if output_path.suffix == ".csv":
                data.to_csv(output_path, index=False)
            elif output_path.suffix == ".json":
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif output_path.suffix == ".zarr":
                data.to_zarr(output_path)
            else:
                raise ValueError(f"Unsupported file extension: {output_path.suffix}")
        elif save_method == "csv":
            data.to_csv(output_path, index=False)
        elif save_method == "json":
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif save_method == "zarr":
            data.to_zarr(output_path)
        
        logger.info(f"Saved data to: {output_path}")
        
        # Save metadata if provided
        if metadata:
            metadata_path = output_path.with_suffix('.metadata.json')
            enhanced_metadata = {
                "file_info": {
                    "filename": output_path.name,
                    "created_at": datetime.now().isoformat(),
                    "file_size_bytes": output_path.stat().st_size if output_path.exists() else None
                },
                "processing_config": self.config.model_dump(),
                **metadata
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(enhanced_metadata, f, indent=2, default=str)
            
            logger.info(f"Saved metadata to: {metadata_path}")
        
        return output_path
    
    def create_summary_report(self,
                            output_files: list[Path],
                            summary_data: Dict,
                            report_name: str = "processing_summary") -> Path:
        """Create a summary report of processed files."""
        summary_dir = self.config.output.base_output_dir / "reports"
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = summary_dir / f"{report_name}_{timestamp}.json"
        
        report_data = {
            "summary": summary_data,
            "output_files": [str(f) for f in output_files],
            "generated_at": datetime.now().isoformat(),
            "configuration": self.config.model_dump()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Created summary report: {summary_path}")
        return summary_path
    
    def list_outputs(self, 
                    variable: Optional[str] = None,
                    region: Optional[str] = None,
                    scenario: Optional[str] = None,
                    output_type: Optional[str] = None) -> list[Path]:
        """List existing output files matching criteria."""
        base_dir = self.config.output.base_output_dir
        
        if not base_dir.exists():
            return []
        
        # Build search pattern
        pattern_parts = []
        if output_type:
            pattern_parts.append(output_type)
        if variable:
            pattern_parts.append(variable.lower())
        if region:
            pattern_parts.append(region.lower())
        if scenario:
            pattern_parts.append(scenario.lower())
        
        # Search for files
        if pattern_parts:
            search_dir = base_dir / Path(*pattern_parts)
            if search_dir.exists():
                return list(search_dir.rglob("*"))
        else:
            return list(base_dir.rglob("*"))
        
        return []
    
    def clean_old_outputs(self, 
                         days_old: int = 30,
                         dry_run: bool = True) -> list[Path]:
        """Clean up old output files."""
        base_dir = self.config.output.base_output_dir
        
        if not base_dir.exists():
            return []
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
        old_files = []
        
        for file_path in base_dir.rglob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                old_files.append(file_path)
                if not dry_run:
                    file_path.unlink()
                    logger.info(f"Deleted old file: {file_path}")
        
        if dry_run and old_files:
            logger.info(f"Found {len(old_files)} files older than {days_old} days (dry run)")
        
        return old_files


def get_output_manager(config: Optional[ClimateConfig] = None) -> OutputManager:
    """Get a configured output manager instance."""
    return OutputManager(config)


def standardize_output_path(variable: str,
                          region: str,
                          scenario: str = "historical",
                          output_type: str = "stats",
                          file_extension: str = "csv",
                          custom_suffix: Optional[str] = None,
                          threshold: Optional[float] = None,
                          config: Optional[ClimateConfig] = None) -> Path:
    """Convenience function to get standardized output path."""
    manager = get_output_manager(config)
    return manager.get_output_path(
        variable=variable,
        region=region,
        scenario=scenario,
        output_type=output_type,
        file_extension=file_extension,
        custom_suffix=custom_suffix,
        threshold=threshold
    )


def ensure_output_directory(output_path: Union[str, Path]) -> Path:
    """Ensure output directory exists and return the path."""
    output_path = Path(output_path)
    output_dir = output_path.parent if output_path.is_file() else output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir 