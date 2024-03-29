find /home/WujieAITeam/private -type d -name "stable-diffusion-xl-base-1.0"

find /home/WujieAITeam/private -name "mjhq30k_imgs.zip"

进程查看 ps -ef   
tmux kill-session -t edm2  
tmux attach -t edm2   
tmux new-session -s edm2   


通过以下命令查看僵尸进程    
sudo fuser -v /dev/nvidia*  
找到COMMAND=python的，然后通过以下命令逐一kill僵尸进程    
sudo kill -9 进程q


huggingface-cli download --repo-type dataset --resume-download playgroundai/MJHQ-30K --local-dir playgroundai/MJHQ-30K
