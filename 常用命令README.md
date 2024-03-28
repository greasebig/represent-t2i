进程查看 ps -ef   
tmux kill-session -t edm2  
tmux attach -t edm2   
tmux new-session -s edm2   


通过以下命令查看僵尸进程    
sudo fuser -v /dev/nvidia*  
找到COMMAND=python的，然后通过以下命令逐一kill僵尸进程    
sudo kill -9 进程q



