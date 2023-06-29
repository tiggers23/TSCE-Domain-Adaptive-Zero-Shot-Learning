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

Acknowledgements: Our codes are mainly based on https://github.com/junbaoZHUO/UODTN