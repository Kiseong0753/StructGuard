# Baselines
CUDA_VISIBLE_DEVICES=0 python main_unlearn_cifar10.py  --num_adv_images 20  --base_pgd_eps 4.0  --baseline_lr 0.001 --reg_lamb 1.0 --seed 0
# Ours
CUDA_LAUNCH_BLOCKING=1 python main_unlearn_cifar10.py  --weight_level 'Fisher' --constraint_type 'sim' --num_adv_images 20   --pgd_eps 3.0 --unlearn_lr 0.005 --projection_lr 0.009 --reg_lamb 1.0 --seed 0 --log_name "CIFAR10" --num_class 10  --device "cuda:1"
