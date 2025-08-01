#!/bin/bash
# monitor_pipeline.sh - Monitor the vLLM pipeline status

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/pipeline_config.sh" ]]; then
    source "$SCRIPT_DIR/pipeline_config.sh"
else
    echo "ERROR: pipeline_config.sh not found!"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REFRESH_INTERVAL=5

clear_screen() {
    printf "\033[2J\033[H"
}

get_job_info() {
    local job_id=$1
    squeue -j "$job_id" -h -o "%T|%N|%l|%M" 2>/dev/null || echo "NOT_FOUND||||"
}

format_time() {
    local time_str=$1
    if [[ -z "$time_str" ]]; then
        echo "N/A"
    else
        echo "$time_str"
    fi
}

show_status() {
    clear_screen
    
    echo -e "${BLUE}=== vLLM Pipeline Monitor ===${NC}"
    echo -e "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Find recent jobs
    HOST_JOBS=$(squeue -u $USER -n "host_llama_dual" -h -o "%i|%T|%N|%l|%M" 2>/dev/null)
    ANALYSIS_JOBS=$(squeue -u $USER -n "parallel_analysis" -h -o "%i|%T|%N|%l|%M|%K" 2>/dev/null)
    CLEANUP_JOBS=$(squeue -u $USER -n "cleanup_vllm" -h -o "%i|%T|%N|%l|%M" 2>/dev/null)
    
    # Host job status
    echo -e "${YELLOW}HOST JOB STATUS:${NC}"
    if [[ -n "$HOST_JOBS" ]]; then
        while IFS='|' read -r job_id state node time_limit time_used; do
            case $state in
                RUNNING) color=$GREEN ;;
                PENDING) color=$YELLOW ;;
                *) color=$RED ;;
            esac
            echo -e "  Job $job_id: ${color}$state${NC} on $node (${time_used}/${time_limit})"
            
            # Check vLLM server status if running
            if [[ "$state" == "RUNNING" && -n "$node" ]]; then
                echo -n "    Servers: "
                active_count=0
                for (( port=$BASE_PORT; port<$((BASE_PORT + NUM_GPUS)); port++ )); do
                    if timeout 2 curl -s "http://${node}:${port}/health" >/dev/null 2>&1; then
                        echo -n "●"
                        ((active_count++))
                    else
                        echo -n "○"
                    fi
                done
                echo " ($active_count/$NUM_GPUS active)"
            fi
        done <<< "$HOST_JOBS"
    else
        echo -e "  ${RED}No host jobs found${NC}"
    fi
    echo ""
    
    # Analysis job status
    echo -e "${YELLOW}ANALYSIS JOB STATUS:${NC}"
    if [[ -n "$ANALYSIS_JOBS" ]]; then
        # Count by state
        running_count=$(echo "$ANALYSIS_JOBS" | grep -c "RUNNING" || true)
        pending_count=$(echo "$ANALYSIS_JOBS" | grep -c "PENDING" || true)
        completed_count=$(echo "$ANALYSIS_JOBS" | grep -c "COMPLETED" || true)
        failed_count=$(echo "$ANALYSIS_JOBS" | grep -c "FAILED" || true)
        
        echo -e "  Running: ${GREEN}$running_count${NC} | Pending: ${YELLOW}$pending_count${NC} | Completed: ${BLUE}$completed_count${NC} | Failed: ${RED}$failed_count${NC}"
        
        # Show first few running jobs
        echo "$ANALYSIS_JOBS" | grep "RUNNING" | head -3 | while IFS='|' read -r job_id state node time_limit time_used array_id; do
            echo -e "  Job $job_id: Task $array_id on $node (${time_used}/${time_limit})"
        done
    else
        echo -e "  ${RED}No analysis jobs found${NC}"
    fi
    echo ""
    
    # Cleanup job status
    echo -e "${YELLOW}CLEANUP JOB STATUS:${NC}"
    if [[ -n "$CLEANUP_JOBS" ]]; then
        while IFS='|' read -r job_id state node time_limit time_used; do
            case $state in
                COMPLETED) color=$GREEN ;;
                PENDING) color=$YELLOW ;;
                RUNNING) color=$BLUE ;;
                *) color=$RED ;;
            esac
            echo -e "  Job $job_id: ${color}$state${NC}"
        done <<< "$CLEANUP_JOBS"
    else
        echo -e "  No cleanup jobs scheduled"
    fi
    echo ""
    
    # Resource usage summary
    echo -e "${YELLOW}RESOURCE SUMMARY:${NC}"
    echo -n "  GPU Usage: "
    squeue -u $USER -o "%b" -h | grep -o "gpu:[0-9]*" | awk -F: '{sum+=$2} END {print sum " GPUs allocated"}'
    
    # Check for output files
    echo ""
    echo -e "${YELLOW}RECENT OUTPUT FILES:${NC}"
    if [[ -d "logs" ]]; then
        ls -lt logs/analysis_*.log 2>/dev/null | head -3 | while read -r line; do
            echo "  $line"
        done
    fi
    
    echo ""
    echo -e "${BLUE}Press Ctrl+C to exit${NC}"
}

# Main monitoring loop
echo "Starting pipeline monitor..."
echo "Refreshing every $REFRESH_INTERVAL seconds..."

while true; do
    show_status
    sleep $REFRESH_INTERVAL
done