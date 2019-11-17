#!/bin/bash
#------------------------------------------------------------------------------
# Script to run competition test case.
#------------------------------------------------------------------------------

function main
{
  test_case="$1"

  if [ "$test_case" != 1 -a "$test_case" != 2 -a "$test_case" != 3 -a \
       "$test_case" != 4 -a "$test_case" != 0 ] ; then
    echo "Usage: ${0##*/} <case>, where <case> = 1, 2, 3 or 4." 1>&2
    exit 1
  fi

  if [ ${USE_GPU:-YES} = YES ] ; then
    exec=exec.gpu
  else
    exec=exec.cpu
  fi

  # one that can be done in a couple minutes as a test, one at 20 minutes, one at 1-2 hours, and one at 6+ hours
  case $test_case in
    0) num_vector=4000 num_field=90000 num_iterations=1 num_proc=2 ;;
    1) num_vector=4000 num_field=90000 num_iterations=400 num_proc=1 ;;
    2) num_vector=18000 num_field=90000 num_iterations=350 num_proc=2 ;;
    3) num_vector=27000 num_field=90000 num_iterations=1600 num_proc=3 ;;
    4) num_vector=54000 num_field=90000 num_iterations=3000 num_proc=6 ;;
  esac

#  if [ "${LMOD_SYSTEM_NAME:-}" = summit -o "${LMOD_SYSTEM_NAME:-}" = peak ] ; then # Summit, Peak
  if [ -n "${LSF_BINDIR:-}" ] ; then
    module load cuda
    launch_cmd="jsrun --nrs $num_proc --rs_per_host $num_proc --bind packed:7 \
                      --cpu_per_rs 7 --gpu_per_rs 1 --tasks_per_rs 1 -X 1"
  elif [ "${HOSTNAME/-*}" = rhea ] ; then
    module load cuda
    launch_cmd="srun -n $num_proc"
  else # gpusys2
    MPI_ROOT="$(spack location --install-dir openmpi)"
    launch_cmd="env $MPI_ROOT/bin/mpirun -n $num_proc --oversubscribe"
  fi

  $launch_cmd ./$exec --num_vector $num_vector \
    --num_field $num_field --num_iterations $num_iterations
}

#------------------------------------------------------------------------------

main $@

#------------------------------------------------------------------------------
