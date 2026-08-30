#!/usr/bin/env bash
# Full experimental campaign, in order, one experiment at a time.
#
# Timing experiments must never overlap: they share the 8 cores of the machine, and a
# laptop under sustained load throttles. Two rules follow from that and are applied
# below:
#   --interleave  puts the repetition in the outer loop, so a slow drift over the
#                 campaign hits every configuration equally instead of penalising
#                 whichever one runs last;
#   --cooldown N  idles N seconds between configurations.
# Budget: about 1 h on an 8-core MacBook. Raw CSVs land in results/.
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/env.sh
cd src
LOG=../results/run_all.log
run(){ echo "===== $* @ $(date +%H:%M:%S) =====" | tee -a $LOG
       python3 -u experiments.py "$@" >>$LOG 2>/dev/null || echo "FAILED: $*" | tee -a $LOG; }

echo "campaign started $(date)" > $LOG

# --- correctness first: nothing else means anything if this fails ---------------
python3 check_correctness.py --dataset ml-100k --k 8 --iters 5 --cores 4 \
        2>/dev/null | tee ../results/correctness.txt
run stats

# --- scalability ---------------------------------------------------------------
run strong       --dataset ml-10m --iters 4 --reps 3 --interleave
run strong       --dataset ml-1m  --iters 4 --reps 3 --interleave --cooldown 15
run factors      --dataset amazon-movies --cores 8 --iters 4 --reps 3 --reps-max-k 256 \
                 --ks 8,16,32,64,128,256 --with-als --als-max-k 128 --interleave --cooldown 20
run factors      --dataset ml-1m --cores 8 --iters 4 --reps 3 --reps-max-k 256 \
                 --ks 8,16,32,64,128,256 --with-als --als-max-k 128 --interleave --cooldown 20

# --- accuracy ------------------------------------------------------------------
run convergence  --dataset amazon-movies --cores 8 --iters 30 --k 32
run convergence  --dataset ml-1m         --cores 8 --iters 30 --k 32
run quality      --dataset amazon-movies --cores 8 --k 32 --als-iters 1 2 3 5 8 12 20 30
run quality      --dataset ml-1m         --cores 8 --k 32 --als-iters 1 2 3 5 8 12 20 30

echo "campaign finished $(date)" | tee -a $LOG
echo "now run ./scripts/build_report.sh"
