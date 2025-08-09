#!/bin/bash
# Optimized batch processing script for generating county climate statistics
# from Zarr files using the modern modular architecture with parallel processing
#
# This script provides several predefined configurations for running
# the optimized batch county statistics processing with parallel support.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_highlight() {
    echo -e "${CYAN}[OPTIMIZED]${NC} $1"
}

# Function to check if required directories exist
check_prerequisites() {
    if [ ! -d "climate_outputs/zarr" ]; then
        print_error "Missing: climate_outputs/zarr directory"
        exit 1
    fi
    
    if [ ! -d "regional_counties" ]; then
        print_error "Missing: regional_counties directory"
        exit 1
    fi
    
    if [ ! -d "src/climate_zarr" ]; then
        print_error "Missing: src/climate_zarr directory"
        exit 1
    fi
    
    print_success "Prerequisites verified"
}

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTION] [--parallel] [--workers N]"
    echo ""
    echo "Processing Options:"
    echo "  all           Process all regions (CONUS, Alaska, Hawaii, Guam, Puerto Rico)"
    echo "  conus         Process CONUS only"
    echo "  alaska        Process Alaska only"
    echo "  hawaii        Process Hawaii only"
    echo "  guam          Process Guam only"
    echo "  puerto_rico   Process Puerto Rico only"
    echo "  custom        Run with custom parameters (interactive)"
    echo ""
    echo "Performance Options:"
    echo "  --parallel    Enable multiprocessing across regions"
    echo "  --workers N   Max total worker processes (default: all CPUs)"
    echo ""
    echo "Other Options:"
    echo "  test          Test optimized batch processor initialization"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 all                           # Process all regions sequentially"
    echo "  $0 all --parallel                # Process all regions in parallel"
    echo "  $0 all --parallel --workers 4    # Use 4 multiprocessing workers"
    echo "  $0 conus                         # Process CONUS only"
    echo "  $0 custom                        # Interactive mode"
    echo ""
}

# Function to parse additional arguments
parse_args() {
    USE_PARALLEL=0
    CPU_COUNT=$(python - <<'PY'
import os
print(os.cpu_count() or 32)
PY
)
    MAX_WORKERS=$CPU_COUNT
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --parallel)
                USE_PARALLEL=1
                shift
                ;;
            --workers)
                MAX_WORKERS="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    export USE_PARALLEL
    export MAX_WORKERS
}

# Function to build command arguments
build_command_args() {
    local regions="$1"
    local scenario="${2:-ssp370}"
    local output="$3"
    
    local cmd_args=(
        "--zarr-dir" "climate_outputs/zarr"
        "--regions" "$regions"
        "--scenario" "$scenario"
        "--output" "$output"
    )
    
    # Add parallel processing args if enabled
    if [ "$USE_PARALLEL" -eq 1 ]; then
        cmd_args+=("--parallel")
        cmd_args+=("--max-workers" "$MAX_WORKERS")
    fi
    
    echo "${cmd_args[@]}"
}

# Function to run the batch processing
run_processing() {
    local regions="$1"
    local scenario="${2:-ssp370}"
    local output="${3:-climate_county_stats_$(date +%Y%m%d_%H%M%S).csv}"
    
    print_status "Processing: $regions (scenario: $scenario)"
    [ "$USE_PARALLEL" -eq 1 ] && print_highlight "Region pool enabled (max processes: $MAX_WORKERS). Counties use all CPUs per region automatically."
    
    # Build and execute command
    local cmd_args
    cmd_args=($(build_command_args "$regions" "$scenario" "$output"))
    
    uv run python batch_county_stats.py "${cmd_args[@]}"
    
    if [ $? -eq 0 ]; then
        print_success "Complete: $output"
        
        # Display file info if available
        if [ -f "$output" ]; then
            local file_size=$(du -h "$output" | cut -f1)
            local line_count=$(wc -l < "$output")
            print_status "Output: $file_size, $((line_count - 1)) records"
        fi
    else
        print_error "Processing failed"
        exit 1
    fi
}

# Function for interactive custom processing
run_custom() {
    echo "Available regions: conus, alaska, hawaii, guam, puerto_rico"
    read -p "Regions (comma-separated): " regions
    
    read -p "Scenario [ssp370]: " scenario
    scenario=${scenario:-ssp370}
    
    read -p "Output filename [climate_county_stats_custom.csv]: " output
    output=${output:-climate_county_stats_custom.csv}
    
    read -p "Enable parallel processing? [y/N]: " enable_parallel
    if [[ $enable_parallel =~ ^[Yy]$ ]]; then
        USE_PARALLEL=1
        read -p "Workers [32]: " workers
        MAX_WORKERS=${workers:-32}
    fi
    
    run_processing "$regions" "$scenario" "$output"
}

# Function to process all regions
process_all() {
    local regions="conus,alaska,hawaii,guam,puerto_rico"
    local output="climate_county_stats_all_regions_$(date +%Y%m%d_%H%M%S).csv"
    run_processing "$regions" "ssp370" "$output"
}

# Function to process single region
process_region() {
    local region="$1"
    local output="climate_county_stats_${region}_$(date +%Y%m%d_%H%M%S).csv"
    run_processing "$region" "ssp370" "$output"
}

# Function to test optimized processor
test_optimized() {
    uv run python -c "import sys; sys.path.append('src'); from batch_county_stats import OptimizedBatchCountyProcessor; print('✅ Processor ready')"
}

# Main script logic
main() {
    # Parse additional arguments first
    parse_args "$@"
    
    # Check prerequisites
    check_prerequisites
    
    case "${1:-help}" in
        "all")
            print_status "Processing all regions..."
            process_all
            ;;
        "conus")
            print_status "Processing CONUS region..."
            process_region "conus"
            ;;
        "alaska")
            print_status "Processing Alaska region..."
            process_region "alaska"
            ;;
        "hawaii")
            print_status "Processing Hawaii region..."
            process_region "hawaii"
            ;;
        "guam")
            print_status "Processing Guam region..."
            process_region "guam"
            ;;
        "puerto_rico")
            print_status "Processing Puerto Rico region..."
            process_region "puerto_rico"
            ;;
        "custom")
            run_custom
            ;;
        "test")
            test_optimized
            ;;
        "help"|*)
            show_usage
            exit 0
            ;;
    esac
}

# Print banner
echo "=========================================="
echo "  Climate County Statistics Processor"
echo "=========================================="
echo ""

# Run main function with all arguments
main "$@"