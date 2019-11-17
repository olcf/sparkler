#!/bin/bash
#------------------------------------------------------------------------------

#if [ "${LMOD_SYSTEM_NAME:-}" = summit ] ; then # Summit
if [ -n "${LSF_BINDIR:-}" ] ; then
  module -q load gcc cuda essl
  #make -f Makefile.summit distclean
  make -f Makefile.summit USE_GPU=${USE_GPU:-YES}
elif [ "${HOSTNAME/-*}" = rhea ] ; then
  module -q load cuda openblas
  #make -f Makefile.rhea distclean
  make -f Makefile.rhea USE_GPU=${USE_GPU:-YES}
else
  #make distclean
  make USE_GPU=${USE_GPU:-YES}
fi

#------------------------------------------------------------------------------
