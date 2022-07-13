set height 0
set pagination off
set logging file gdb.txt
set logging overwrite on
set logging on
#set trace-commands on
set print frame-arguments all

rbreak ^\(vtkOSPRayVolumeNode\|vtkOSPRayUnstructuredVolumeMapperNode\)::
commands
silent
backtrace 3
echo \n
continue
end
