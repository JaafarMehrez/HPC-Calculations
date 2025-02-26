#!/bin/bash
#SBATCH --account=def-nike-ab
#SBATCH --mem=1950G
#SBATCH --time=2-23:30
#SBATCH --cpus-per-task=12
#SBATCH --signal=B:SIGUSR1@18000
#SBATCH --output=out.%j
#SBATCH --error=err.%j

module load StdEnv/2023
module load intel/2023.2.1
module load openmpi/4.1.5

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

APP_OUT="$SLURM_TMPDIR/dmrcc.out"
APP_ERR="$SLURM_TMPDIR/dmrcc.err"

# Signal handler to copy files and exit
function sig_handler_USR1() {
    echo "SIGUSR1 received at $(date) - copying critical files"
    
    # Terminate dmrcc if still running
    if kill -0 $DM_PID 2>/dev/null; then
        echo "Killing dmrcc (PID: $DM_PID)"
        kill $DM_PID
        wait $DM_PID 2>/dev/null
    fi
    
    # Copy files back to submit directory
    rsync --include=fort.16 --exclude=fort.* -v "$SLURM_TMPDIR"/* "$SLURM_SUBMIT_DIR/"
    exit 0
}

trap sig_handler_USR1 SIGUSR1

# Copy input files to node-local storage
rsync -r * "$SLURM_TMPDIR/"
cd "$SLURM_TMPDIR"
echo "Starting calculation at: $(date)"

# Launch dmrcc in the background and capture PID
dmrcc > "$APP_OUT" 2> "$APP_ERR" &
DM_PID=$!


sleep 10000000 &
wait
