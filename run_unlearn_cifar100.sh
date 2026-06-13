# Baselines
CUDA_VISIBLE_DEVICES=0 python main_unlearn_cifar100.py --num_adv_images 200  --base_pgd_eps 4.0  --baseline_lr 0.001 --reg_lamb 1.0 --seed 0 
# Ours
CUDA_VISIBLE_DEVICES=0  python main_unlearn_cifar100.py --weight_level 'Fisher' --constraint_type 'sim' --num_adv_images 200 --pgd_eps 1.0  --unlearn_lr 0.001 --projection_lr 0.01 --reg_lamb 1.0 --seed 0 --log_name "CIFAR100" --num_class 100  --device "cuda:0"
