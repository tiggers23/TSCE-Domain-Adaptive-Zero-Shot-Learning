# Three-way Semantic Consistent Embedding
Code release for Semantic Consistent Embedding for Domain Adaptive Zero-Shot Learning

# Prepare
Please follow https://github.com/junbaoZHUO/UODTN to get the dataset.

Please run
'python train_gcn_basic_awa_ezhuo_2019.py'
in https://github.com/junbaoZHUO/UODTN for getting the GCN model.

Note that different from the original UODTN fitting both weights and bias of the classification layer, we only fit the weights when training the GCN model.
# Training TSCE
Please run
'python train_AwA.py -gpu <gpu_id>'
for training our model.

# Citation
Please cite our paper:

@ARTICLE{10183844,
  author={Zhang, Jianyang and Yang, Guowu and Hu, Ping and Lin, Guosheng and Lv, Fengmao},
  journal={IEEE Transactions on Image Processing}, 
  title={Semantic Consistent Embedding for Domain Adaptive Zero-Shot Learning}, 
  year={2023},
  volume={32},
  number={},
  pages={4024-4035},
  doi={10.1109/TIP.2023.3293769}}


Acknowledgements: Our codes are mainly based on https://github.com/junbaoZHUO/UODTN