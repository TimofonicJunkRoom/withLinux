Add Condor Jobs
---

1. code preparation

2. htcondor universe

```
standard -- no parallel, need to re-link with `condor_compile`
vanilla -- programs cannot be relinked, e.g. scripts.
grid
java
scheduler -- for lightweight, immediate jobs
local -- for different conditions of the job, job will never be preempted
parallel -- euch as MPI jobs
vm
docker
```

3. submit description file

```
# example 1
executable = myexe
log        = myexe.log
input      = myexe.input
output     = myexe.output
queue
```

```
# example 2
executable = foo
universe   = standard
log        = foo.log
queue
# input and output points to /dev/null
```

```
# example 3
executable = mathematica
universe   = vanilla
input      = inputfile
output     = outputfile
error      = errorfile
log        = logfile
request_memory = 1 GB

initialdir = run_1
queue

initialdir = run_2
queue
```

```
arguments = ....
output = xxx.out.$(OpSys).$(Arch).$(Process)
queue xxx
request_GPUs  = 1
machine_count = 8 # universe: parallel
request_cpus  = 8 # universe:parallel
```

4. submit job

After Jobs submitted
---

* use `condor_rm JOB_ID` to remove jobs.

* run `condor_q -analyze JOB_ID` to investigate why a job is not running.

