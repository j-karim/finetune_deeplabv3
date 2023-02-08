# syncs the repo to a lambda gpu instance

src_base_dir=~/Documents/pycharm_projects/Dida
dst_base_dir=Dida
instance_ip=192.9.133.121

echo $src_base_dir/* ubuntu@$instance_ip:~/$dst_base_dir
rsync -av --exclude 'checkpoints' --exclude 'logs' --exclude 'results' $src_base_dir/* ubuntu@$instance_ip:$dst_base_dir

ssh ubuntu@$instance_ip chmod +x /home/ubuntu/Dida/TrainScript.py
echo chmod +x /home/ubuntu/Dida/TrainScript.py
exit

