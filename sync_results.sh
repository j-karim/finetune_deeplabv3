# syncs the results from lambda gpu instance back to local

instance_ip=192.9.133.121

src_base_dir_1=Dida/checkpoints
dst_dir=~/Documents/pycharm_projects/Dida

src_base_dir_2=Dida/logs
rsync -av --exclude '**/*/00*' ubuntu@$instance_ip:/home/ubuntu/$src_base_dir_1 $dst_dir
rsync -av ubuntu@$instance_ip:/home/ubuntu/$src_base_dir_2 $dst_dir



