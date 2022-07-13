
define BacktraceTo
    set $BacktraceTo_OldFile = $_gdb_setting_str("logging file")

    set logging off
    eval "set logging file %s", "$arg0"
    set logging on

    bt

    set logging off
    eval "set logging file %s", $BacktraceTo_OldFile
    set logging on
end 

set height 0
set pagination off
set logging file gdb.txt
set logging overwrite on
set logging on
set trace-commands on

start

br vtkMPICommunicatorMPIErrorHandler
commands
    up-silently 999
    set $_rank = opt_rank
    p $_rank

    eval "BacktraceTo gdb_%lu.txt", $_rank

    cont
end
