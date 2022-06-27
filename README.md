# Reproducible Example: VTK + MPI + DistributedDataFilter (D3) Error

I experienced a error that could be consistently reproduced with the code in
this repo. The error I got looks like:

```console
$ mpirun -np 6 vtkPDistributedDataFilterExample
...
Generic Warning: In /home/thobson/src/vtkOSPRay/external/vtk/Parallel/MPI/vtkMPICommunicator.cxx, line 64
MPI had an error
------------------------------------------------
Message truncated, error stack:
internal_Recv(127).......: MPI_Recv(buf=0x555556a351a0, count=21, MPI_INT, 0, 1020203, MPI_COMM_WORLD, status=0x7fffffff8444) failed
MPIDIG_recv_type_init(72): Message from rank 0 and tag 2 truncated; 84 bytes received but buffer size is 88
------------------------------------------------

Abort(469377038) on node 1 (rank 1 in comm 0): application called MPI_Abort(MPI_COMM_WORLD, 469377038) - process 1
```

The rest of this README covers how to reproduce this error.

**Note:** The setup is based on my scripts that are intended to build everything
*from scratch. But, I expect that any VTK + MPI + OSPRay (optional) setup should
*work to reproduce the error.


## Setup

Everything is packaged into a script, so all the commands to build things are
just calls into that script. A command like `./go.sh foo bar` will run bash
functions within the `go.sh` script, first `go` then `go-foo` and then
`go-foo-bar`. The final function executed is the one that actually calls other
executables. For example, `go-docker-build` calls `docker build` with the right
arguments.

Side-note: This repository was originally for testing VTK + OSPRay and then
later morphed into VTK + MPI + OSPRay, hence why the build steps include a
superfluous OSPRay build.

```console
$ ./go.sh docker build
$ ./go.sh docker start
$ ./go.sh spack git clone
$ ./go.sh spack env create
$ time ./go.sh spack install
...
==> Updating view at /home/thobson/src/VTK-MPI-Error-ReprEx/senv/.spack-env/view

real    26m49.339s
user    0m0.138s
sys     0m0.072s
$ ./go.sh ospray git clone
$ ./go.sh ospray cmake configure
$ time ./go.sh ospray cmake parbuild
...

real    9m59.172s
user    0m0.116s
sys     0m0.153s
$ ./go.sh ospray cmake install
$ ./go.sh vtk git clone
$ ./go.sh vtk cmake configure
$ time ./go.sh vtk cmake parbuild
...

real    5m54.225s
user    0m0.121s
sys     0m0.225s
$ ./go.sh vtk cmake install
$ ./go.sh src cmake configure
$ time ./go.sh src cmake build
...

real    0m15.056s
user    0m0.026s
sys     0m0.025s
$ ./go.sh src cmake install
```

## Reproducing the Error

```console
$ ./go.sh src exec mpirun -np 3 vtkPDistributedDataFilterExample
...

Generic Warning: In /home/thobson/src/VTK-MPI-Error-ReprEx/external/vtk/Parallel/MPI/vtkMPICommunicator.cxx, line 64
MPI had an error
------------------------------------------------
Message truncated, error stack:
internal_Recv(127).......: MPI_Recv(buf=0x56499d53afd0, count=21, MPI_INT, 0, 1020203, MPI_COMM_WORLD, status=0x7ffed3e21054) failed
MPIDIG_recv_type_init(72): Message from rank 0 and tag 2 truncated; 84 bytes received but buffer size is 88
------------------------------------------------

Generic Warning: In /home/thobson/src/VTK-MPI-Error-ReprEx/external/vtk/Parallel/MPI/vtkMPICommunicator.cxx, line 64
MPI had an error
------------------------------------------------
Message truncated, error stack:
internal_Recv(127).......: MPI_Recv(buf=0x5599a5266f80, count=21, MPI_INT, 0, 1020203, MPI_COMM_WORLD, status=0x7fffee065374) failed
MPIDIG_recv_type_init(72): Message from rank 0 and tag 2 truncated; 84 bytes received but buffer size is 88
------------------------------------------------

Abort(133832718) on node 2 (rank 2 in comm 0): application called MPI_Abort(MPI_COMM_WORLD, 133832718) - process 2
Abort(1073356814) on node 1 (rank 1 in comm 0): application called MPI_Abort(MPI_COMM_WORLD, 1073356814) - process 1
```
