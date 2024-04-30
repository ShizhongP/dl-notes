
filename=$1
echo "shell  PID: $$"
echo "output file: $filename"
echo "If you want kill the process, execute 'kill -9 \$(cat nvidia-smi.pid)'"
nohup nvidia-smi -l 1 --format=csv --query-gpu=timestamp,name,index,utilization.gpu,memory.total,memory.used,power.draw >> $filename &
echo $! > ./nvidia-smi.pid 

