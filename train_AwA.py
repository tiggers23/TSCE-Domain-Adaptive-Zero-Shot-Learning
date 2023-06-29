# -*- coding: utf-8 -*-
import os
import sys
from gcn_models.gcn import GCN

sys.path.append('./')
import torch
import json
import torch.nn.functional as F
import argparse
from dataset import load_images
from model_I2WebV import CDZS
from tqdm import tqdm
import numpy as np
import getpass
from nltk.corpus import wordnet as wn


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer


def compute_accuracy(test_loader, class_list, device, model, test_classes=None):
    with torch.no_grad():
        model.eval()
        class_dic = {j: i for i, j in enumerate(class_list)}
        if test_classes is not None:
            unpredic_classes = []
            for name in class_list:
                if name not in test_classes:
                    unpredic_classes.append(name)
            # unpredic_classes = list(set(class_list) - set(test_classes))
            unpredic_classes_id = []
            for i in unpredic_classes:
                unpredic_classes_id.append(class_dic[i])
        else:
            unpredic_classes_id = None
        # fetch attributes

        predict_labels_total = []
        re_batch_labels_total = []

        for samples in tqdm(test_loader):
            x = samples["image"].to(device)
            y = samples["label"].to(device)
            batch_size = y.shape[0]
            sample_features = model(x, y, is_train='Validation')
            if unpredic_classes_id is not None:
                sample_features[:, unpredic_classes_id] -= 100
            _, predict_labels = torch.max(sample_features, 1)
            predict_labels = predict_labels.cpu().numpy()
            true_labels = y.cpu().numpy()

            predict_labels_total = np.append(predict_labels_total, predict_labels)
            re_batch_labels_total = np.append(re_batch_labels_total, true_labels)

        # compute averaged per class accuracy
        predict_labels_total = np.array(predict_labels_total, dtype='int')
        re_batch_labels_total = np.array(re_batch_labels_total, dtype='int')
        unique_labels = np.unique(re_batch_labels_total)
        acc = 0
        num = 0
        acc_1 = 0
        # print("class num: {}".format(unique_labels.shape[0]))
        for l in unique_labels:
            idx = np.nonzero(re_batch_labels_total == l)[0]

            # acc += accuracy_score(re_batch_labels_total[idx], predict_labels_total[idx])

            acc_class = np.sum(predict_labels_total[idx] == l) / idx.shape[0]
            acc_1 += acc_class

            acc += np.sum(predict_labels_total[idx] == l)
            num += idx.shape[0]
        acc_1 = acc_1 / unique_labels.shape[0]
        acc = acc / num
        model.train()
        return acc, acc_1


def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))


def save_best(zsl_acc, gzsl_unseen_acc, gzsl_seen_acc, gzsl_h, i, best_results):
    if zsl_acc > best_results['best_zsl_acc'][0]:
        best_results['best_zsl_acc'][0] = zsl_acc
        best_results['best_zsl_acc'][1] = i
    if gzsl_unseen_acc > best_results['best_gzsl_unseen_acc'][0]:
        best_results['best_gzsl_unseen_acc'][0] = gzsl_unseen_acc
        best_results['best_gzsl_unseen_acc'][1] = i
    if gzsl_seen_acc > best_results['best_gzsl_seen_acc'][0]:
        best_results['best_gzsl_seen_acc'][0] = gzsl_seen_acc
        best_results['best_gzsl_seen_acc'][1] = i
    if gzsl_h > best_results['best_gzsl_h'][0]:
        best_results['best_gzsl_h'][0] = gzsl_h
        best_results['best_gzsl_h'][1] = i
    return best_results


def main():
    parser = argparse.ArgumentParser(description="CDZS Training")

    parser.add_argument(
        "--dataset",
        type=str,
        default="I2AwA",
        choices=["I2AwA", "I2WebV"],
        help="dataset name (default: Office-31)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "--entropy_weight",
        type=float,
        default=5,
        help="weight of entropy loss",
    )
    parser.add_argument(
        "--distribute_weight",
        type=float,
        default=5,
        help="weight of max entropy loss",
    )
    parser.add_argument(
        "--fc_w",
        type=float,
        default=5,
        help="weight of fc loss",
    )
    parser.add_argument('--text', type=str, default='word2vec')
    parser.add_argument('--distribute_bound', type=float, default=100.0)
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument(
        "--item",
        type=int,
        default=2000,
        help="number of training items",
    )

    parser.add_argument(
        "--test_item",
        type=int,
        default=1999,
        help="the time of test",
    )
    parser.add_argument(
        "--consistency_type",
        type=str,
        default="pair",
        help="the type of consistency loss, entropy/l2",
    )
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=10,
        help="the weight of consistency loss",
    )

    parser.add_argument(
        "--max_warehoure",
        type=int,
        default=768,
        help="the size of warehoure",
    )

    parser.add_argument('--temperature', type=float, default=0.05,
                        help='temperature (default: 0.05)')

    parser.add_argument(
        "--gpu",
        type=str,
        default='0',
        help="the device location",
    )
    parser.add_argument('--bias_weight', type=float, default=0.3,
                        help='temperature (default: 0.05)')
    args = parser.parse_args()

    dataset_dic = {'I2AwA': {'source': '3D2-1', 'target': 'AwA'},
                   'I2WebV': {'source': 'imagenet', 'target': 'webvision'}}

    json_file_dic = {'I2AwA': "/data0/home/zhangjianyang/cdzs_data/awa2-split.json",
                     'I2WebV': "/data0/home/zhangjianyang/cdzs_data/web_wnids.json"}

    with open(json_file_dic[args.dataset], 'r') as file:
        classes_json = json.load(file)
    args.seen_classes = classes_json['train_names']
    args.unseen_classes = []
    for i in classes_json['test_names']:
        if i not in args.seen_classes:
            args.unseen_classes.append(i)

    wordnet2name = {}

    k = 0
    for i, j in enumerate(classes_json['train_names']):
        wordnet2name[classes_json['train'][i]] = j
    for i, j in enumerate(classes_json['test_names']):
        wordnet2name[classes_json['test'][i]] = j

    args.source_domain = dataset_dic[args.dataset]['source']
    args.target_domain = dataset_dic[args.dataset]['target']

    print(f'source_domain: {args.source_domain}, target_domain: {args.target_domain}')
    args.source_target_domain = [args.source_domain, args.target_domain]
    save_dir_0 = 'experiment'
    save_dir_1 = args.dataset
    save_dir_2 = args.source_target_domain[0] + '-' + args.source_target_domain[1]
    save_name = 'gcnwarm_{}_withwarehousebanlence{}_entropy_{}_consis_{}'.format(args.fc_w,
                                                                                 args.max_warehoure,
                                                                                 args.entropy_weight,
                                                                                 args.consistency_weight,
                                                                                 )
    print(save_name)
    print(args.use_warehouse)

    save_dir = os.path.join(".", save_dir_0, save_dir_1, save_dir_2, save_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = "cuda:{}".format(args.gpu)
    args.device = device
    data_dim = {'I2AwA': "/data0/home/zhangjianyang/cdzs_data/",
                'I2WebV': '/data0/home/zhangjianyang/cdzs_data/'
                }

    source_dim = data_dim[args.dataset] + args.source_target_domain[0]
    target_dim = data_dim[args.dataset] + args.source_target_domain[1]

    graph = json.load(open('/data0/home/zhangjianyang/UODTN/GCN/materials/AWA2/animals_graph_all.json'))

    wnids = graph['wnids']
    n = len(wnids)
    edges = graph['edges']
    edges = edges + [(v, u) for (u, v) in edges]
    edges = edges + [(u, u) for u in range(n)]

    word_vectors = torch.tensor(graph['vectors']).to(device)
    word_vectors = F.normalize(word_vectors)
    model = CDZS(300, word_vectors, args)
    model.Embedding = GCN(n, edges, word_vectors.shape[1], 2048, 'd2048,d', device)
    model.embeddings = word_vectors
    model.Embedding.load_state_dict(
        torch.load("/data0/home/zhangjianyang/UODTN/GCN/RESULTS_MODELS/awa-basic3/epoch-3000.pth"))
    model = model.to(device)

    parameter_list = [{"params": model.CNN.layer3.parameters(), "lr": 1},
                      {"params": model.CNN.layer4.parameters(), "lr": 2}]
    optimizer = torch.optim.SGD(parameter_list, lr=1, momentum=0.9, weight_decay=0.0001, nesterov=True)

    optimizer_c = torch.optim.SGD(model.Embedding.parameters(), lr=5, momentum=0.9, weight_decay=0.0001, nesterov=True)
    tbar = tqdm(range(args.item))

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    param_lr_c = []
    for param_group in optimizer_c.param_groups:
        param_lr_c.append(param_group["lr"])
    best_zsl_acc = [0, 0]
    best_gzsl_h = [0, 0]
    best_gzsl_seen_acc = [0, 0]
    best_gzsl_unseen_acc = [0, 0]
    best_results = {"best_zsl_acc": best_zsl_acc,
                    "best_gzsl_h": best_gzsl_h,
                    "best_gzsl_seen_acc": best_gzsl_seen_acc,
                    "best_gzsl_unseen_acc": best_gzsl_unseen_acc
                    }
    model.train()
    class_list = []
    name2iid = {}
    seen_classes_id = []
    unseen_classes_id = []
    for k, wn in enumerate(wnids[127:177]):
        class_name = wordnet2name[wn]
        class_list.append(class_name)
        name2iid[class_name] = k
        if class_name in args.seen_classes:
            seen_classes_id.append(k)
        elif class_name in args.unseen_classes:
            unseen_classes_id.append(k)

    print(f'number of seen classes: {len(seen_classes_id)}')
    print(f'number of unseen classes: {len(unseen_classes_id)}')
    train_source_loader = load_images(source_dim, class_list, name2iid, args=args, batch_size=args.batch_size,
                                      split="train",
                                      unseen_classes=args.unseen_classes)
    train_target_loader = load_images(target_dim, class_list, name2iid, batch_size=args.batch_size,
                                      split="train_target")
    test_target_seen_loader = load_images(target_dim, class_list, name2iid, batch_size=args.batch_size, split="val",
                                          unseen_classes=args.unseen_classes)
    test_target_unseen_loader = load_images(target_dim, class_list, name2iid, batch_size=args.batch_size, split="val",
                                            unseen_classes=args.seen_classes)
    train_source_iter = iter(train_source_loader)
    train_target_iter = iter(train_target_loader)

    for i in tbar:
        if ((i + 0) % args.test_item) == 0:
            zsl_acc, zsl_acc_1 = compute_accuracy(test_target_unseen_loader, class_list, device, model,
                                                  args.unseen_classes)
            gzsl_unseen_acc, gzsl_unseen_acc_1 = compute_accuracy(test_target_unseen_loader, class_list, device, model)
            gzsl_seen_acc, gzsl_seen_acc_1 = compute_accuracy(test_target_seen_loader, class_list, device, model)
            gzsl_h = 2 * (gzsl_unseen_acc * gzsl_seen_acc) / (gzsl_unseen_acc + gzsl_seen_acc)
            best_results = save_best(zsl_acc, gzsl_unseen_acc, gzsl_seen_acc, gzsl_h, i, best_results)

            print(
                "epoch: {}\n zsl_acc: {:.2%}, gzsl_unseen_acc: {:.2%}, gzsl_seen_acc: {:.2%}, gzsl_unseen_class: {:.2%}, gzsl_seen_class: {:.2%}".format(
                    i,
                    zsl_acc,
                    gzsl_unseen_acc,
                    gzsl_seen_acc,
                    gzsl_unseen_acc_1,
                    gzsl_seen_acc_1))

        try:
            source_sample = train_source_iter.__next__()
        except:
            train_source_iter = iter(train_source_loader)
            source_sample = train_source_iter.__next__()

        try:
            target_sample = train_target_iter.__next__()
        except:
            train_target_iter = iter(train_target_loader)
            target_sample = train_target_iter.__next__()

        optimizer = inv_lr_scheduler([1, 2], optimizer, i, 0.001, 0.75, init_lr=args.lr)
        optimizer_c = inv_lr_scheduler([5], optimizer_c, i, 0.001, 0.75, init_lr=args.lr)
        x_s = source_sample['image'].to(device)
        y_s = source_sample['label'].to(device)
        x_t = target_sample['image'].to(device)
        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss, source_loss, structure_loss = model(x_s, y_s, seen_label=seen_classes_id, is_train='True')
        target_loss, entropy_loss, dis_loss, bl_loss = model(x_t, seen_label=unseen_classes_id, is_train='Target')
        target_loss = target_loss + loss
        target_loss.backward()
        optimizer.step()
        optimizer_c.step()
        tbar.set_description(
            "source_loss: {:.3f}, target: {:.3f}, bl: {:.3f}".format(loss.item(), target_loss.item() - loss.item(),
                                                                     bl_loss))

    print(
        "best_zsl_acc: {0[0]} on batch {0[1]}, best_gzsl_unseen_acc: {1[0]} on batch {1[1]}, best_gzsl_seen_acc: {2[0]} on batch {2[1]}, best_gzsl_h: {3[0]} on batch {3[1]}".format(
            best_results["best_zsl_acc"],
            best_results["best_gzsl_unseen_acc"],
            best_results["best_gzsl_seen_acc"],
            best_results["best_gzsl_h"]))

    with open(os.path.join(save_dir, 'result.txt'), 'w') as f:
        f.write(args.source_target_domain[0] + '-' + args.source_target_domain[
            1] + '\n' + 'maxentropy_withoutwarehouse_entropy_{}_maxentropy_{}_{}_consistency_0'.format(
            args.entropy_weight, args.distribute_weight, args.distribute_bound) + '\n')
        f.write(
            "epoch: {}\n zsl_acc: {:.2%}, gzsl_unseen_acc: {:.2%}, gzsl_seen_acc: {:.2%}, gzsl_unseen_class: {:.2%}, gzsl_seen_class: {:.2%}".format(
                i,
                zsl_acc,
                gzsl_unseen_acc,
                gzsl_seen_acc,
                gzsl_unseen_acc_1,
                gzsl_seen_acc_1))
        f.write('\n')
        f.write(
            "best_zsl_acc: {0[0]} on batch {0[1]}, best_gzsl_unseen_acc: {1[0]} on batch {1[1]}, best_gzsl_seen_acc: {2[0]} on batch {2[1]}, best_gzsl_h: {3[0]} on batch {3[1]}".format(
                best_results["best_zsl_acc"],
                best_results["best_gzsl_unseen_acc"],
                best_results["best_gzsl_seen_acc"],
                best_results[
                    "best_gzsl_h"]))


if __name__ == '__main__':
    main()
