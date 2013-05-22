echo "Compiling:"
/usr/lib64/openmpi/bin/mpicc "$@" -Wall -pedantic -Wno-long-long -o aquadPartA aquadPartA.c stack.h stack.c

if [ $? -ne 0 ]; then
  echo "Error while compiling."
  exit 1
fi

echo "Running:"
/usr/lib64/openmpi/bin/mpirun -c 5 ./aquadPartA
