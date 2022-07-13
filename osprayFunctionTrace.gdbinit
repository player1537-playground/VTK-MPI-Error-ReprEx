set height 0
set pagination off
set logging file gdb.txt
set logging overwrite on
set logging on
#set trace-commands on
set print frame-arguments all

rbreak ^osp
commands
silent
backtrace
echo \n
continue
end
