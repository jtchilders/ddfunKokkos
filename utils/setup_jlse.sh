

#####
# setup CONDA environment for python
# if CONDA_HOME not set, use a default
if [ -z "$CONDA_HOME" ]; then
   export CONDA_HOME=/vast/projects/datascience/parton/conda/2025-01
   echo "Using default CONDA_HOME: $CONDA_HOME"
fi
echo "Setup conda environment for python in: $CONDA_HOME"
source $CONDA_HOME/bin/activate

######
# modules to use for building
echo "Loading modules for cuda/12.3.0 gcc/12.2.0 cmake/3.28.3"
module load cuda/12.3.0 gcc/12.2.0 cmake/3.28.3

######
# setup Kokkos

# if KOKKOS_ARCH not set, report error and exit
if [ -z "$KOKKOS_ARCH" ]; then
   echo "KOKKOS_ARCH not set"
   echo "   OPTIONS: KOKKOS_ARCH=Kokkos_ARCH_VOLTA70; KOKKOS_ARCH=Kokkos_ARCH_AMPERE80;"
   return 0
fi

# if KOKKOS_VERSION not set, use default
if [ -z "$KOKKOS_VERSION" ]; then
   export KOKKOS_VERSION=4.5.01
fi
# same for KOKKOS_BUILD
if [ -z "$KOKKOS_BUILD" ]; then
   export KOKKOS_BUILD=Release
fi

# construct KOKKOS_HOME
export KOKKOS_HOME=/vast/projects/datascience/parton/kokkos/kokkos-$KOKKOS_VERSION/$KOKKOS_ARCH/$KOKKOS_BUILD

export INSTPATH=install
export KOKKOS_PREFIX=$KOKKOS_HOME/kokkos/$INSTPATH
export CMAKE_PREFIX_PATH=$KOKKOS_PREFIX/lib64/cmake/Kokkos
export LD_LIBRARY_PATH=$KOKKOS_PREFIX/lib64:$LD_LIBRARY_PATH