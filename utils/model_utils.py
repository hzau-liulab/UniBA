import random
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
# from itertools import repeat, product
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score, average_precision_score, \
    matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Dataset as GeoDataset
# from transformers import get_linear_schedule_with_warmup
from collections import defaultdict
# from model.vae import *
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.pdb_utils import *
import math
import json
# from model.CGNet import CGNet
# from model.RESNet import RESNet
from openpyxl import load_workbook

torch.autograd.set_detect_anomaly(True)


def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False


def get_pr_list(args, task=None, fold=-1):
    if fold == -1:
        protein_chain = np.atleast_1d(
            np.loadtxt(f'./data/split_data/{args.split_level}/{task}_pair.txt', dtype=str)
        )
        return protein_chain
    else:
        tr_pr_list = np.atleast_1d(
            np.loadtxt(f'./data/split_data/{args.split_level}/train_cv{fold + 1}.txt', dtype=str)
        )
        val_pr_list = np.atleast_1d(
            np.loadtxt(f'./data/split_data/{args.split_level}/val_cv{fold + 1}.txt', dtype=str)
        )
        return tr_pr_list, val_pr_list


def get_mut_pr_list(args, task=None, fold=-1):
    if fold == -1:
        protein_chain = np.loadtxt(f'./data/split_data/{args.split_level}_mut/{task}_pair.txt', dtype=str)
        return protein_chain
    else:
        tr_pr_list = np.loadtxt(f'./data/split_data/{args.split_level}_mut/train_cv{fold + 1}.txt', dtype=str)
        val_pr_list = np.loadtxt(f'./data/split_data/{args.split_level}_mut/val_cv{fold + 1}.txt', dtype=str)
        return tr_pr_list, val_pr_list


def init_nested_dict(task_mode, types):
    task_keys = {"reg": ["reg"], "cla": ["cla"], "both": ["reg", "cla"]}[task_mode]
    return {t: {tp: [] for tp in types} for t in task_keys}


def append_outputs(all_outputs, all_labels, key, pred, true, task_mode="reg"):
    if task_mode == "cla":
        # pred = torch.sigmoid(pred)  # [B]
        # pred = pred.view(-1)
        pred = torch.softmax(pred, dim=-1)  # [B, C]
        true = true.view(-1)  # [B]
    else:  # reg
        pred = pred.view(-1)  # [B]
        true = true.view(-1)  # [B]

    for p, t in zip(pred, true):
        all_outputs[task_mode][key].append(p.detach().cpu())
        all_labels[task_mode][key].append(t.detach().cpu().item())


def compute_str_loss(out_dict, label):
    y_true = label["pKd"]
    y_pred = out_dict["pKd"]   # 或 "pKd"
    y_res = out_dict.get("y_res", None)
    y_cg = out_dict.get("y_cg", None)
    cg_gate = out_dict.get("cg_gate")

    total_loss = F.mse_loss(y_pred, y_true)
    if y_res is None or y_cg is None or cg_gate is None:
        return total_loss

    err_res = (y_res - y_true).abs().detach()
    err_cg = (y_cg - y_true).abs().detach()
    diff = err_res - err_cg
    margin = 0.5
    target_gate = torch.sigmoid(diff / margin)

    gain = torch.relu(diff)
    weight = torch.sqrt(gain + 1e-6)
    weight = (weight / (weight.mean() + 1e-6))
    weight = weight.clamp(max=3.0)
    gate_loss = F.binary_cross_entropy(cg_gate, target_gate, reduction="none")
    gate_loss = (gate_loss * weight.detach()).mean()

    # confidence = torch.abs(target_gate - 0.5) * 2  # 0~1
    # gate_loss = F.binary_cross_entropy(cg_gate, target_gate, reduction="none")
    # gate_loss = (gate_loss * confidence).mean()

    total_loss = total_loss + 0.05 * gate_loss

    return total_loss


def compute_loss(out_dict, label):

    y_true = label["pKd"].reshape(-1)

    y_pred = out_dict["pKd"].reshape(-1)
    y_seq = out_dict["y_seq"]
    y_str = out_dict["y_str"]
    seq_gate = out_dict["seq_gate"]

    main_loss = F.mse_loss(y_pred, y_true)

    loss_seq = (y_seq - y_true).pow(2)
    loss_str = (y_str - y_true).pow(2)

    margin = 0.5
    gate_target = torch.sigmoid((loss_str - loss_seq) / margin)
    weight = torch.sigmoid(3.0 * (torch.abs(y_seq - y_str).detach() - 0.5))
    gate_loss = F.binary_cross_entropy(seq_gate.reshape(-1), gate_target.reshape(-1)) * weight

    entropy = -(seq_gate * torch.log(seq_gate + 1e-8) +
                (1 - seq_gate) * torch.log(1 - seq_gate + 1e-8))

    total_loss = main_loss + 0.03 * gate_loss - 0.005 * entropy

    return total_loss

# def compute_loss(out_dict, label):
#     y_true = label["pKd"]
#     y_pred = out_dict["pKd"]   # 或 "pKd"
#     y_seq = out_dict.get("y_seq", None)
#     y_str = out_dict.get("y_str", None)
#     seq_gate = out_dict.get("seq_gate", None)
#
#     total_loss = F.mse_loss(y_pred, y_true)
#
#     pred_seq_err = out_dict.get("pred_seq_err", None)
#     pred_str_err = out_dict.get("pred_str_err", None)
#
#     if (
#         pred_seq_err is None or
#         pred_str_err is None or
#         y_seq is None or
#         y_str is None
#     ):
#         return total_loss
#
#     true_seq_err = torch.abs(y_seq - y_true).detach()
#     true_str_err = torch.abs(y_str - y_true).detach()
#
#     loss_err = (
#         F.mse_loss(pred_seq_err, true_seq_err) +
#         F.mse_loss(pred_str_err, true_str_err)
#     )
#
#     diff = true_str_err - true_seq_err
#     margin = 0.9
#     target_gate = torch.sigmoid(diff / margin)
#     gain = torch.abs(diff)
#
#     weight = torch.sqrt(gain + 1e-6)
#     weight = weight / (weight.mean() + 1e-6)
#     weight = weight.clamp(max=3.0)
#
#     gate_loss = F.binary_cross_entropy(seq_gate, target_gate, reduction="none")
#     gate_loss = (gate_loss * weight.detach()).mean()
#
#     total_loss = total_loss + 0.1 * loss_err + 0.1 * gate_loss
#
#     return total_loss
#

def mse_scaled(pred, true, scale=1.0):
    pred = pred.squeeze()
    true = true.squeeze()
    return F.mse_loss(pred, true) / scale   #.to(pred.device)


def loss_fn(out_dict, target, lambda_gate=0.01, lambda_resp=0.1):

    pred = out_dict["pred"]
    gate_logits = out_dict["gate_logits"]
    gate_weights = out_dict["gate_weights"]
    expert_preds = out_dict["expert_preds"]

    # 1. 主损失
    loss_main = F.mse_loss(pred, target)

    # 2. gating 熵正则
    gate_prob = F.softmax(gate_logits, dim=-1)
    loss_gate = - (gate_prob * torch.log(gate_prob + 1e-8)).sum()

    # 3. 责任损失（仅被选 expert）
    loss_resp = 0.0
    for k in expert_preds:
        loss_resp += gate_weights[k] * F.mse_loss(
            expert_preds[k], target
        )

    return loss_main + lambda_gate * loss_gate + lambda_resp * loss_resp


def mse_expert_weighted(out_dict, target, device=None, lambda_resp=1.0, lambda_anchor=0.2):
    expert_preds = out_dict["expert_preds"]
    expert_weights = out_dict["expert_weights"]

    loss_main = torch.zeros((), device=device)
    loss_resp = torch.zeros((), device=device)
    loss_anchor = torch.zeros((), device=device)

    if device is None:
        device = target.device

    for name in expert_preds:
        pKd_k = expert_preds[name].to(device)
        w_k = expert_weights[name].to(device)

        err_sq = (pKd_k - target) ** 2
        # 1. 主损失：影响最终 pKd 的最优性
        loss_main = loss_main + w_k * err_sq

        # 2. 责任损失：被信任时必须拟合
        loss_resp = loss_resp + w_k.detach() * err_sq

        # 3. 锚定损失：防止专家塌缩（等价于单独训练的 MSE）
        loss_anchor = loss_anchor + err_sq

    return loss_main + lambda_resp * loss_resp + lambda_anchor * loss_anchor


# ---- load seq embedding ----
def load_all_seq_embeddings(
    root="./feature/seq_features",
    plm='mint'
):
    mint_emb = {}

    for fname in os.listdir(root):
        if not fname.endswith(f"_{plm}_embeddings.pt"):
            continue

        # e.g. ppi_mut_mint_embeddings.pt
        name = fname.replace(f"_{plm}_embeddings.pt", "")
        parts = name.split("_")

        if len(parts) == 1:
            data_type = parts[0]      # ppi / aai / tcr-pmhc
            key = "wt"
        else:
            data_type, key = parts    # ppi + mut

        path = os.path.join(root, fname)
        print(f"[Load] {path}")

        emb_dict = torch.load(path, map_location="cpu")

        mint_emb.setdefault(data_type, {})[key] = emb_dict

    return mint_emb


def build_label_dict(pair, data_dict, device, interface_dict=None):
    is_mut = "." in pair
    pair_wt = pair.split(".")[0]
    wt_entry = data_dict[pair_wt]
    data_type = wt_entry["type"]
    dt = f"{data_type}_mut" if is_mut else data_type

    if is_mut:
        mut_idx = wt_entry["mut_ids"].index(pair)
        label = {
            "pKd": torch.tensor(wt_entry["pKd_mut"][mut_idx], device=device).squeeze(0),
            "label_aff_cls": torch.tensor(wt_entry['label_mut'][mut_idx], device=device),
            "label_cls": torch.tensor(1, device=device),
            # "ddG": torch.tensor(wt_entry["ddG"][mut_idx], device=device),
            # "Temperature": torch.tensor(wt_entry["temperature"][mut_idx], device=device),
        }
    else:
        label = {
            "pKd": torch.tensor(wt_entry["pKd_wt"], device=device).squeeze(0),
            # "label_aff_cls": torch.tensor(wt_entry["label_wt"], device=device),
            "label_cls": torch.tensor(0, device=device),
        }

    iface = None
    if interface_dict:
        iface = interface_dict[pair]

    return label, wt_entry, iface


def get_seq_embedding(pair, wt_entry, mint_emb, device):
    is_mut = "." in pair
    key = "mut" if is_mut else "wt"
    data_type = wt_entry["type"]  #   "c-met-AF3"
    return mint_emb[data_type][key][pair].to(device), key


def get_str_graph(args, pair, wt_entry, graph_type):
    is_mut = "." in pair
    key = "mut" if is_mut else "wt"

    data_type = wt_entry["type"]
    sub_dir = f"{data_type}_mut" if is_mut else data_type

    if graph_type == "cg_intra":
        parts = pair.split(".", 1)
        complex_id = parts[0]
        mut_info = parts[1] if is_mut else None

        try:
            pdb, chain1, chain2 = complex_id.split("_")
        except ValueError:
            raise ValueError(f"Invalid pair format for cg_intra: {pair}")

        base_dir = f"{args.data_path}/{graph_type}_graph/{sub_dir}"
        if mut_info is None:
            pc1_graph_file = f"{base_dir}/{pdb}_{chain1}.gh"
            pc2_graph_file = f"{base_dir}/{pdb}_{chain2}.gh"
        else:
            pc1_graph_file = f"{base_dir}/{pdb}_{chain1}_{mut_info}.gh"
            pc2_graph_file = f"{base_dir}/{pdb}_{chain2}_{mut_info}.gh"
        #     return None, key  # no wt cg_intra graph

        if os.path.exists(pc1_graph_file) and os.path.exists(pc2_graph_file):
            return (
                torch.load(pc1_graph_file),
                torch.load(pc2_graph_file),
            ), key

        return None, key

    graph_file = f"{args.data_path}/{graph_type}_graph/{sub_dir}/{pair}.gh"  #{sub_dir}

    if os.path.exists(graph_file):
        graph = torch.load(graph_file)
        graph.data_type = data_type
        return graph, key

    return None, key


def weighted_sample_batches(all_pr_list, batch_size, num_batches,
                            tr_pr_list=None, tr_mut_pr_list=None,
                            seed=None):
    """
    按加权随机采样生成 batch 列表。
    返回: list of list，每个子列表就是一个 batch
    """
    rng = np.random.default_rng(seed)

    if tr_pr_list is None or tr_mut_pr_list is None:
        weights = np.ones(len(all_pr_list)) / len(all_pr_list)
    else:
        wt_set = set(tr_pr_list)
        mut_set = set(tr_mut_pr_list)
        weights = []
        for pr in all_pr_list:
            if pr in mut_set:
                weights.append(1.0 / len(mut_set))
            else:
                weights.append(1.0 / len(wt_set))
        weights = np.array(weights)
        weights /= weights.sum()  # 归一化

    batches = []
    for _ in range(num_batches):
        sampled_idx = rng.choice(len(all_pr_list), size=batch_size, replace=True, p=weights)
        batch = [all_pr_list[i] for i in sampled_idx]
        batches.append(batch)

    return batches


def collect_params_by_keys(model, keys):
    return [
        p for name, p in model.named_parameters()
        if name in keys and p.requires_grad
    ]


def load_pretrained_module(model, fold, pre_fold_best_epoch, device='cpu', plm="mint"):
    """
    加载 intra + inter 的预训练参数（按 fold），
    并划分 intra / inter / other 参数。
    """

    model_dict = model.state_dict()
    loaded_keys = {"res": [], "cg": [], "str": [], "seq": []}
    # loaded_keys = {"moe": [], "repr": []}
    # res_unfreeze_keys, cg_unfreeze_keys, seq_unfreeze_keys = [], [], []

    if model.use_res:
        res_epoch = pre_fold_best_epoch["res"][str(fold)]
        # res_path = f"/home/yyShen/NAcontact/ablation/res_wo_hand_str_feat_1792/res_{fold}_{res_epoch}.pth"  # _1layer
        res_path = f"/home/yyShen/NAcontact/model/res_module/res_{fold}_{res_epoch}.pth"  #_1layer
        res_state = torch.load(res_path, map_location=device)

        for k, v in res_state.items():
            new_k = f"res_net.{k}" if not k.startswith("res_net.") else k
            if new_k in model_dict and model_dict[new_k].shape == v.shape:
                model_dict[new_k] = v
                loaded_keys["res"].append(new_k)

    if model.use_cg:
        cg_epoch = pre_fold_best_epoch["cg"][str(fold)]
        cg_path = f"/home/yyShen/NAcontact/model/cg_intra/cg_{fold}_{cg_epoch}.pth"
        cg_state = torch.load(cg_path, map_location=device)

        for k, v in cg_state.items():
            new_k = f"cg_net.{k}" if not k.startswith("cg_net.") else k
            if new_k in model_dict and model_dict[new_k].shape == v.shape:
                model_dict[new_k] = v
                loaded_keys["cg"].append(new_k)

    if model.use_str:
        # str_epoch = pre_fold_best_epoch["str"][str(fold)]
        # # str_path = f"/home/yyShen/NAcontact/ablation/str_wo_hand_str_feat_1792/str_{fold}_{str_epoch}.pth"
        # str_path = f"/home/yyShen/NAcontact/model/str_module/str_{fold}_{str_epoch}.pth"
        # str_state = torch.load(str_path, map_location=device)

        # res_epoch = pre_fold_best_epoch["res"][str(fold)]
        # # res_path = f"/home/yyShen/NAcontact/ablation/res_wo_hand_str_feat_1792/res_{fold}_{res_epoch}.pth"  # _1layer
        # res_path = f"/home/yyShen/NAcontact/model/res_module/res_{fold}_{res_epoch}.pth"  #_1layer
        # res_state = torch.load(res_path, map_location=device)

        cg_epoch = pre_fold_best_epoch["cg"][str(fold)]
        cg_path = f"/home/yyShen/NAcontact/model/cg_intra/cg_{fold}_{cg_epoch}.pth"  #_1layer
        cg_state = torch.load(cg_path, map_location=device)

        for k, v in cg_state.items():
            new_k = f"str_net.{k}" if not k.startswith("str_net.") else k
            if new_k in model_dict and model_dict[new_k].shape == v.shape:
                model_dict[new_k] = v
                loaded_keys["str"].append(new_k)

    if model.use_seq:
        # ===== seq module（新增）=====
        seq_epoch = pre_fold_best_epoch["seq"][str(fold)]
        seq_path = f"/home/yyShen/NAcontact/model/seq_module/{plm}_{fold}_{seq_epoch}.pth"
        seq_state = torch.load(seq_path, map_location=device)

        for k, v in seq_state.items():
            new_k = f"seq_net.{k}" if not k.startswith("seq_net.") else k
            if new_k in model_dict and model_dict[new_k].shape == v.shape:
                model_dict[new_k] = v
                loaded_keys["seq"].append(new_k)

    # if model.ensemble_mode in ["moe", "fusion"]:
    #
    #     out_epoch = pre_fold_best_epoch["moe"][str(fold)]
    #     out_path = f"/home/yyShen/NAcontact/model/UniAffinity/UniBA_moe_{fold}_{out_epoch}.pth"
    #
    #     out_state = torch.load(out_path, map_location=device)
    #
    #     for k, v in out_state.items():
    #         # new_k = f"output_head.{k}" if not k.startswith("output_head.") else k
    #
    #         if k in model_dict and model_dict[k].shape == v.shape:
    #             model_dict[k] = v
    #             loaded_keys.setdefault("moe", []).append(k)
    #
    # if model.ensemble_mode in ["repr", "fusion"]:
    #
    #     repr_epoch = pre_fold_best_epoch["repr"][str(fold)]
    #     repr_path = f"/home/yyShen/NAcontact/model/UniAffinity/UniBA_repr_{fold}_{repr_epoch}.pth"
    #
    #     repr_state = torch.load(repr_path, map_location=device)
    #
    #     for k, v in repr_state.items():
    #         # new_k = f"repr_head.{k}" if not k.startswith("repr_head.") else k
    #
    #         if k in model_dict and model_dict[k].shape == v.shape:
    #             model_dict[k] = v
    #             loaded_keys.setdefault("repr", []).append(k)

    model.load_state_dict(model_dict)
    backbone_keys = loaded_keys["res"] + loaded_keys["cg"] \
                    + loaded_keys["str"] + loaded_keys["seq"]
                    #  + loaded_keys["moe"] + loaded_keys["repr"]

    other_keys = [
                      k for k in model_dict.keys() if k not in backbone_keys
                  ]      # + res_unfreeze_keys + cg_unfreeze_keys + seq_unfreeze_keys

    # backbone_keys = list(set(backbone_keys) - set(res_unfreeze_keys) - set(cg_unfreeze_keys) - set(seq_unfreeze_keys))

    key_dict = {
        "res": loaded_keys["res"],
        "cg": loaded_keys["cg"],
        "str": loaded_keys["str"],
        "seq": loaded_keys["seq"],
        "backbone": backbone_keys,
        "fusion": other_keys
    }

    return model, key_dict

    # # ============================
    # # 1. intra 模块
    # # ============================
    # intra_epoch = pre_fold_best_epoch['intra'][str(fold)]
    # intra_path = (
    #     f"{args.modules_path}/intra_module2/intra_reg_{fold}_{intra_epoch}.pth"
    # )
    # intra_state = torch.load(intra_path, map_location=device)
    #
    # intra_filtered = {
    #     k: v for k, v in intra_state.items()
    #     if k in model_dict and v.shape == model_dict[k].shape
    # }
    #
    # # ============================
    # # 2. inter 模块
    # # ============================
    # inter_epoch = pre_fold_best_epoch['inter'][str(fold)]
    # inter_path = (
    #     f"{args.modules_path}/inter_module/inter_reg_{fold}_{inter_epoch}.pth"
    # )
    # inter_state = torch.load(inter_path, map_location=device)
    #
    # inter_filtered = {
    #     k: v for k, v in inter_state.items()
    #     if k in model_dict and v.shape == model_dict[k].shape
    # }
    #
    # # ============================
    # # 3. 更新模型
    # # ============================
    # model_dict.update(intra_filtered)
    # model_dict.update(inter_filtered)
    # model.load_state_dict(model_dict)
    #
    # # ============================
    # # 4. 参数划分
    # # ============================
    # intra_keys = list(intra_filtered.keys())
    # inter_keys = list(inter_filtered.keys())
    # other_keys = [
    #     k for k in model_dict.keys()
    #     if k not in intra_keys and k not in inter_keys
    # ]
    #
    # key_dict = {
    #     'intra': intra_keys,
    #     'inter': inter_keys,
    #     'other': other_keys
    # }
    # #
    # # # ============================
    # # # 5. 设置微调参数（保持你原有接口）
    # # # ============================
    # # pretrained_keys = set(intra_keys) | set(inter_keys)
    # # model.set_finetune(model, pretrained_keys, finetune=True)
    #
    # return model, key_dict


def load_pretrained_model(args, model, device, fold=None, pre_model='AAI'):
    """
    统一加载预训练参数，并分类 intra/inter/other。

    pre_model 支持：
        'AAI'       - 迁移 PPI → AAI
        'PPINet'    - PPINet 自身微调
        'AAI_self'  - AAI 自身微调
    """
    model_dict = model.state_dict()

    if pre_model == 'AAI':
        # === 迁移 PPI → AAI ===
        # intra_model = CGNet(input_dim=25, hidden_dims=[256, 256],
        #                     num_relation=5, edge_input_dim=51, num_angle_bin=6,
        #                     graph_construction_model=cg_gh_model, task_type='ppi')
        # inter_model = RESNet(ab_in_dim=1792, ag_in_dim=1792, hidden_dims=[512], task_type='ppi')
        intra_path = f'{args.modules_path}/intra_module/ppi_RESNet5_47.pth'
        inter_path = f'{args.modules_path}/inter_module/ppi_intra_cla_11.pth'
        intra_state = torch.load(intra_path, map_location=device)
        inter_state = torch.load(inter_path, map_location=device)

        pre_state_dict = torch.load(f'{args.modules_path}/{args.classifier}_model/PPINet.pth',
                                    map_location=device)

        matched_params = {k: v for k, v in pre_state_dict.items()
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched_params)
        model.load_state_dict(model_dict)

        intra_keys = set(intra_model.state_dict().keys())
        inter_keys = set(inter_model.state_dict().keys())
        other_keys = [k for k in model_dict.keys() if k not in pre_state_dict.keys()]

        key_dict = {
            'intra': [k for k in matched_params if k in intra_keys],
            'inter': [k for k in matched_params if k in inter_keys],
            'other': other_keys
        }

    elif pre_model in ['PPINet', 'AAINet']:
        # === PPINet 自身微调 或 AAI 自身微调 ===
        if pre_model == 'PPINet':
            resnet_path = f'{args.modules_path}/{args.classifier}_model/ppi_inter2_6.pth'
            cgnet_path = f'{args.modules_path}/{args.classifier}_model/ppi_intra_cla_11.pth'
        else:  # AAI_self
            resnet_path = f'{args.modules_path}/{args.classifier}_model/aai_RESNet4_{fold}.pth'
            cgnet_path = f'{args.modules_path}/{args.classifier}_model/aai_CGNet2_{fold}.pth'

        resnet_state = torch.load(resnet_path, map_location=device)
        cgnet_state = torch.load(cgnet_path, map_location=device)

        cgnet_filtered = {k: v for k, v in cgnet_state.items()
                          if k in model_dict and v.shape == model_dict[k].shape}
        resnet_filtered = {k: v for k, v in resnet_state.items()
                           if k in model_dict and v.shape == model_dict[k].shape}

        model_dict.update(cgnet_filtered)
        model_dict.update(resnet_filtered)
        model.load_state_dict(model_dict)

        intra_keys = set(cgnet_filtered.keys())
        inter_keys = set(resnet_filtered.keys())
        other_keys = [k for k in model_dict.keys() if k not in intra_keys and k not in inter_keys]

        key_dict = {
            'intra': list(intra_keys),
            'inter': list(inter_keys),
            'other': other_keys
        }

    # === 设置微调参数 ===
    pretrained_keys = set(key_dict['intra']) | set(key_dict['inter'])
    model.set_finetune(model, pretrained_keys, finetune=True)

    return model, key_dict


def to_ordinal_label(label_true):
    """
    label_true: scalar tensor {0,1,2}
    return: tensor shape [2]
    """
    if label_true == 0:      # low
        return torch.tensor([0., 0.])
    elif label_true == 1:    # mid
        return torch.tensor([1., 0.])
    elif label_true == 2:    # high
        return torch.tensor([1., 1.])


def ordinal_logits_to_probs(logits):
    """
    logits: Tensor [B, 2]  (ordinal logits)
    return: Tensor [B, 3]  (p_low, p_mid, p_high)
    """
    p_gt_low = torch.sigmoid(logits[:, 0])  # P(y > low)
    p_gt_mid = torch.sigmoid(logits[:, 1])  # P(y > mid)

    p_low = 1.0 - p_gt_low
    p_mid = p_gt_low * (1.0 - p_gt_mid)
    p_high = p_gt_mid

    probs = torch.stack([p_low, p_mid, p_high], dim=-1)  # [B, 3]
    return probs


def load_pretrained_model_(args, model, cg_gh_model, device, fold, pre_model='AAI'):
    """
    加载 PPI 预训练参数到 AffinityNet 模型，并划分为 intra/inter/other 参数。

    Args:
        model: 构造完成的 AffinityNet 模型（未加载参数）
        cg_gh_model: 图构建模块
        device: 设备
        pre_model:预训练模型类型
    Returns:
        model: 已加载参数的模型
        pretrained_key_dict: {'intra': [...], 'inter': [...], 'other': [...]}
    """
    if pre_model == 'AAI':
        intra_model = CGNet(input_dim=25, hidden_dims=[512] * 3,
                            num_relation=5, edge_input_dim=51, num_angle_bin=6,
                            graph_construction_model=cg_gh_model, task_type='ab_ag')
        intra_model_path = f'{args.modules_path}/{args.classifier}_model/aai_CGNet2_{fold}.pth'
        intra_model.load_state_dict(torch.load(intra_model_path, map_location=device))

        inter_model = RESNet(ab_in_dim=1298, ag_in_dim=1810, edge_in_dim=15,
                             hidden_dims=[512], task_type='ab_ag')
        inter_model_path = f'{args.modules_path}/{args.classifier}_model/aai_RESNet4_{fold}.pth'
        inter_model.load_state_dict(torch.load(inter_model_path, map_location=device))

        pre_state_dict = {}
        pre_state_dict.update(intra_model.state_dict())
        pre_state_dict.update(inter_model.state_dict())

    else:
        # === 加载预训练参数 ===
        pre_model_path = f'{args.modules_path}/{args.classifier}_model/PPINet.pth'
        pre_state_dict = torch.load(pre_model_path, map_location=device)

        # === 构建 CGNet / GAMP 原型，用于识别参数 key ===
        intra_model = CGNet(input_dim=25, hidden_dims=[256, 256],
                            num_relation=5, edge_input_dim=51, num_angle_bin=6,
                            graph_construction_model=cg_gh_model, task_type='ppi')

        inter_model = RESNet(ab_in_dim=1792, ag_in_dim=1792, hidden_dims=[512], task_type='ppi')

    intra_keys = set(intra_model.state_dict().keys())
    inter_keys = set(inter_model.state_dict().keys())

    # === 匹配参数并加载 ===
    model_state = model.state_dict()
    matched_params = {k: v for k, v in pre_state_dict.items()
                      if k in model_state and v.shape == model_state[k].shape}
    model_state.update(matched_params)
    model.load_state_dict(model_state)

    # 参数 key 分类
    pretrained_keys_intra = [k for k in matched_params if k in intra_keys]
    pretrained_keys_inter = [k for k in matched_params if k in inter_keys]
    other_keys = [k for k in model_state.keys() if k not in pre_state_dict.keys()]

    # 返回模型与 key 映射字典
    key_dict = {
        'intra': pretrained_keys_intra,
        'inter': pretrained_keys_inter,
        'other': other_keys,
    }

    return model, key_dict


def pearson_corr(pred, target, eps=1e-8):
    pred = (pred - pred.mean()) / (pred.std() + eps)
    target = (target - target.mean()) / (target.std() + eps)
    return torch.mean(pred * target)


def mulit_loss_fnc(pro_output, rna_output, pred_contact, batch_label, target_contact):
    # weight_rna = 1.5
    # weight_contact = 1.5
    # weight_pro = 1.0
    pro_target, rna_target = batch_label
    #
    # loss_bd = F.binary_cross_entropy(pred_contact, target_contact, reduction='none')
    # loss_contact = loss_bd.mean() * weight_contact

    pos_weight = torch.tensor([8.0], device=pro_output.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # loss_fn = nn.BCELoss()
    loss_pro = loss_fn(pro_output.squeeze(-1), pro_target)

    # loss_rna = loss_fn(rna_output.squeeze(-1), rna_target) * weight_rna

    # pro_binding_pred = torch.max(pred_contact, dim=1).values
    # rna_binding_pred = torch.max(pred_contact, dim=0).values
    #
    # loss_pro_binding = loss_fn(pro_binding_pred, pro_target)
    # loss_rna_binding = loss_fn(rna_binding_pred, rna_target)

    loss = loss_pro
    # loss = loss_contact + loss_pro + loss_rna + loss_pro_binding + loss_rna_binding
    return loss


# def contrastive_loss(embeddings, labels, margin):
#     pos_pairs = embeddings[labels == 1]
#     neg_pairs = embeddings[labels == 0]
#
#     pos_distances = F.pairwise_distance(pos_pairs.unsqueeze(1), pos_pairs.unsqueeze(0))
#     neg_distances = F.pairwise_distance(pos_pairs.unsqueeze(1), neg_pairs.unsqueeze(0))
#
#     pos_loss = torch.mean(pos_distances)
#     neg_loss = torch.mean(F.relu(margin - neg_distances))
#
#     return pos_loss + neg_loss


def get_graph_filename(pdb, chain_str, mut_chain=None, mut_info=None):
    """
    根据突变链是否包含在 chain_str 中，生成含突变信息的图文件名。
    chain_str: 可以是 "H_L"（ab图）或 "C"（ag图）
    mut_chain: 实际发生突变的链ID
    mut_info: "H.W98A" 等突变信息
    """
    if mut_chain and mut_info and mut_chain in chain_str.split("_"):
        return f"{pdb}_{chain_str}.{mut_chain}.{mut_info}"
    else:
        return f"{pdb}_{chain_str}"


class GraphDataset(Dataset):
    def __init__(self, args, pr, data_dict, graph_type, data_type):
        super().__init__()

        self.pr_list = [pr]
        self.data_dict = data_dict
        self.cg_graph_dir = f'{args.data_path}/{graph_type}_inter_graph/{data_type}'
        self.res_graph_dir = f'{args.data_path}/{graph_type}_graph/{data_type}'
        self.atom_graph_dir = f'{args.data_path}/{graph_type}_graph/{data_type}'
        self.graph_type = graph_type
        self.data_type = data_type

    def __len__(self):
        return len(self.pr_list)

    def __getitem__(self, idx):
        pr = self.pr_list[idx]
        parts = pr.split('.', 1)
        complex_id = parts[0]
        mut_info = parts[1] if len(parts) > 1 else None
        mut_chains = list({mut[1] for mut in mut_info.split('_')}) if mut_info else []
        pdb, chain1, chain2 = complex_id.split('_')
        chains = {'pc1': list(chain1), 'pc2': list(chain2)}

        if self.data_dict is not None:
            is_mut = 'mut' in self.data_type.lower()
            if is_mut:
                wt_entry = self.data_dict[complex_id]
                mut_entries = wt_entry["mut_ids"]
                mut_idx = mut_entries.index(pr)
                pKd_mut = wt_entry['pKd_mut'][mut_idx]
                ddg_true = wt_entry['ddG'][mut_idx]
                Temperature = wt_entry['temperature'][mut_idx]
                label_mut = wt_entry['label_mut'][mut_idx]
                label_cls = 1
                label_ddg = 1 if ddg_true > 0 else 0
            else:
                wt_entry = self.data_dict[pr]
                pKd_mut = ddg_true = Temperature = label_mut = label_ddg = None
                label_cls = 0

            # pKd_wt = wt_entry['pKd_wt']
            # label_wt = wt_entry['label_wt']
            label = {
                "pKd_wt": wt_entry.get('pKd_wt', None),
                "pKd_mut": pKd_mut,
                "label_wt": wt_entry.get('label_wt', None),
                "label_mut": label_mut,
                "label_cls": label_cls,  # WT=0, MUT=1
                "label_ddg": label_ddg,
                "ddG": ddg_true,
                "Temperature": Temperature,
            }

        else:
            label = {k: None for k in ["pKd_wt", "pKd_mut", "label_wt", "label_mut", "label_ddg", "ddG", "Temperature"]}
            label["label_cls"] = 0

        if self.graph_type == "cg":
            graph_file = f"{self.cg_graph_dir}/{pr}.gh"
        elif self.graph_type == "atom":
            graph_file = f'{self.atom_graph_dir}/{pr}.gh'
        else:
            graph_file = f'{self.res_graph_dir}/{pr}.gh'

        if os.path.exists(graph_file):
            graph_data = torch.load(graph_file)
            # with open(graph_file, 'rb') as fd:
            #     graph_data = pickle.load(fd)
        else:
            graph_data = None

        return graph_data, label, pr

        # graph_files = {
            #     tag: f"{self.graph_dir}/{pdb}_{''.join(chain_group)}"
            #          f"{'.' + mut_info if mut_info and any(c in mut_chains for c in chain_group) else ''}.gh"
            #     for tag, chain_group in chains.items()
            # }
            # pc1_graph_file, pc2_graph_file = graph_files["pc1"], graph_files["pc2"]
            #
            # if os.path.exists(pc1_graph_file) and os.path.exists(pc2_graph_file):
            #     with open(pc1_graph_file, 'rb') as fb:
            #         pc1_graph = pickle.load(fb)
            #     with open(pc2_graph_file, 'rb') as fg:
            #         pc2_graph = pickle.load(fg)
            #     graph_data = (pc1_graph, pc2_graph)
            # else:
            #     seq_feat_file = f"{self.seq_feat_dir}/{pr}.pkl"
            #     with open(seq_feat_file, "rb") as f:
            #         seq_feat = pickle.load(f)
            #     pc1_seq = torch.as_tensor(seq_feat["chain1_seqcoding"], dtype=torch.float32)
            #     pc2_seq = torch.as_tensor(seq_feat["chain2_seqcoding"], dtype=torch.float32)
            #     graph_data = (pc1_seq, pc2_seq)


def collate_fn(batch):
    # batch_size=1，直接取第一个元素
    graph_data, label_dict, pr = batch[0]
    if graph_data is None:
        return None, None, pr
    # 将 label_dict 中每个 key 的内容转换为 tensor
    label_tensor_dict = {}
    for key, value in label_dict.items():
        if value is None:
            # 占位，可以是 nan 或 0，根据 loss 是否会 mask
            label_tensor_dict[key] = torch.tensor([float('nan')], dtype=torch.float32)
        elif isinstance(value, (int, float)):
            label_tensor_dict[key] = torch.tensor([value], dtype=torch.float32)
        elif isinstance(value, list):
            label_tensor_dict[key] = torch.tensor(value, dtype=torch.float32)
        else:
            raise TypeError(f"Unknown type for label_dict[{key}]: {type(value)}")

    return graph_data, label_tensor_dict, pr


def batch_Graphdata(args, batch_pr, data_dict=None, graph_type='res', data_type='aai'):
    dataset = GraphDataset(args, batch_pr, data_dict, graph_type, data_type)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    return data_loader


def get_linear_decay_lambda(base_lr, decay_steps, decay_rate, min_lr):
    def lr_lambda(epoch):
        lr = base_lr * (decay_rate ** (epoch / decay_steps))
        return max(lr, min_lr)

    return lr_lambda


def get_adaptive_decay_lambda(base_lr, total_epochs, min_lr, gamma,
                              min_step=5, mid_step1=20, mid_step2=30, final_step=40):
    def generate_decay_epochs():
        decay_epochs = []
        epoch = 0

        # 前期：前 15%
        while epoch + min_step < total_epochs * 0.1:
            epoch += min_step
            decay_epochs.append(epoch)

        while epoch + mid_step1 < total_epochs * 0.3:
            epoch += mid_step1
            decay_epochs.append(epoch)

        while epoch + mid_step2 < total_epochs * 0.6:
            epoch += mid_step2
            decay_epochs.append(epoch)

        while True:
            next_epoch = epoch + final_step
            if next_epoch <= total_epochs:
                epoch = next_epoch
                decay_epochs.append(epoch)
            else:
                break

        return decay_epochs

    def lr_lambda(epoch):
        progress = epoch / total_epochs
        ratio = (1 - progress) ** 2  # power 控制下降速度
        return max(ratio, min_lr / base_lr)

    return lr_lambda


def get_stagewise_decay_lambda(base_lr, total_epochs, min_lr, gamma, switch_epoch=130):
    decay_epochs = [10, 30, 60, 90, 110, 130, 150, 180]

    num_decays_at_switch = sum(switch_epoch >= e for e in decay_epochs)
    base_ratio = gamma ** num_decays_at_switch

    def lr_lambda(epoch):
        if epoch < switch_epoch:
            num_decays = sum(epoch >= e for e in decay_epochs)
            ratio = gamma ** num_decays
        else:

            progress = (epoch - switch_epoch) / (total_epochs - switch_epoch)
            power_ratio = (1 - progress) ** 2
            ratio = base_ratio * power_ratio

        return max(ratio, min_lr / base_lr)

    return lr_lambda


def get_sinusoidal_lr_lambda(base_lr, sinusoidal_epochs, min_lr):
    def lr_lambda(epoch):
        progress = (epoch + 1) / (sinusoidal_epochs + 1)
        ratio = math.sin(math.pi * progress)  # 半个sin波，从0 -> 1 -> 0
        return max(ratio, min_lr / base_lr)

    return lr_lambda


def get_cosine_annealing_lr_lambda(base_lr, total_epochs, min_lr):
    def lr_lambda(epoch):
        # cosine from 0 to pi => 从 1 降到 -1，然后平移归一化到 1 -> 0
        cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
        return max(cosine_decay, min_lr / base_lr)

    return lr_lambda


def load_scaler(args):
    with open(f'./feature/{args.task}_scaler_tr.pkl', 'rb') as fs:
        scaler_ = pickle.load(fs)
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = scaler_
    return scaler


# def get_pr_list(args, task=None, fold=-1):
#     if fold == -1:
#         protein_chain = np.loadtxt(f'./data/split_data/{args.split_level}/{task}_pair.txt', dtype=str)
#         return protein_chain
#     else:
#         np.random.seed(2025)
#         tr_pr_list = np.loadtxt(f'./data/split_data/{args.split_level}/train_cv{fold + 1}.txt', dtype=str)
#         val_pr_list = np.loadtxt(f'./data/split_data/{args.split_level}/val_cv{fold + 1}.txt', dtype=str)
#         np.random.shuffle(tr_pr_list)
#         return tr_pr_list, val_pr_list





def auto_split_params(args, model, tr_pr_list, data_dict):
    def trace_used_modules(model, fn, *args):
        used = set()
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                h = module.register_forward_hook(lambda m, inp, out, n=name: used.add(n))
                hooks.append(h)
        with torch.no_grad():
            fn(*args)
        for h in hooks:
            h.remove()
        return used

    model.eval()
    pair = tr_pr_list[0]
    pair_wt = pair.split('.')[0]
    data_type = data_dict[pair_wt]["type"]

    intra_data, label_dict, _ = next(iter(
        batch_Graphdata(args, pair, data_dict, "cg", data_type)))
    inter_data, _, _ = next(iter(
        batch_Graphdata(args, pair, data_dict, "res", data_type)))
    pc1_graph, pc2_graph = intra_data
    inter_graph = inter_data

    # ---- tracing ----
    intra_used = trace_used_modules(model, model._process_intra_graph, pc1_graph)
    intra_used |= trace_used_modules(model, model._process_intra_graph, pc2_graph)
    inter_used = trace_used_modules(model, model._process_inter_graph, inter_graph)

    # ---- split params ----
    intra_params, inter_params, other_params = [], [], []

    for name, p in model.named_parameters():
        if any(name.startswith(n) for n in intra_used):
            intra_params.append(p)
        elif any(name.startswith(n) for n in inter_used):
            inter_params.append(p)
        else:
            other_params.append(p)

    model.train()
    return intra_params, inter_params, other_params


def cla_metrics(predicted, y_true, y_score):
    AUC = roc_auc_score(y_true, y_score)
    AUPR = average_precision_score(y_true, y_score)
    Precision = precision_score(y_true, predicted)
    MCC = matthews_corrcoef(y_true, predicted)
    Recall = recall_score(y_true, predicted)
    F1 = f1_score(y_true, predicted)
    return round(Recall, 4), round(Precision, 4), round(F1, 4), \
           round(AUC, 4), round(AUPR, 4), round(MCC, 4)


# def reg_metrics(y_true, y_pred):
#     MAE = mean_absolute_error(y_true, y_pred)
#     RMSE = mean_squared_error(y_true, y_pred) ** 0.5
#     Pearson_corr, _ = pearsonr(y_true, y_pred)
#     Spearman_corr, _ = spearmanr(y_true, y_pred)
#     return round(MAE, 4), round(RMSE, 4), round(Pearson_corr, 4), round(Spearman_corr, 4)


def reg_metrics(y_true, y_pred):
    """
    y_true, y_pred: array-like (np.ndarray / list / torch->numpy)
    """
    # 统一为 numpy（若本来就是 ndarray，不会复制）
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    mask = ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size < 2:
        return (np.nan, np.nan, np.nan, np.nan)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    try:
        pearson = pearsonr(y_true, y_pred)[0]
    except Exception:
        pearson = np.nan

    try:
        spearman = spearmanr(y_true, y_pred)[0]
    except Exception:
        spearman = np.nan

    return (
        round(mae, 4),
        round(rmse, 4),
        round(pearson, 4),
        round(spearman, 4),
    )


def merge_result(results_all, new_result):
    for top_key in ['reg', 'cla']:
        if top_key in new_result:
            for metric_key, metric_value in new_result[top_key].items():
                results_all[top_key][metric_key] = metric_value


def epoch_evaluate(task_name, all_labels, all_outputs, task_type="both"):

    results = {"reg": {}, "cla": {}}

    # 自动获取数据类型
    data_types = list(all_outputs.get("reg", {}).keys() or all_outputs.get("cla", {}).keys())

    for key in data_types:
        # 回归指标
        if task_type in ["reg", "both"] and len(all_outputs["reg"].get(key, [])) > 0:
            preds = torch.stack(all_outputs["reg"][key]).numpy().reshape(-1)
            trues = np.array(all_labels["reg"][key]).reshape(-1)

            mae, rmse, pearson, spearman = reg_metrics(trues, preds)
            results["reg"][f"{key}_{task_name}"] = {
                "MAE": mae,
                "RMSE": rmse,
                "Pearson": pearson,
                "Spearman": spearman
            }

        # 分类指标
        if task_type in ["cla", "both"] and len(all_outputs["cla"].get(key, [])) > 0:
            preds = torch.stack(all_outputs["cla"][key]).numpy()   # [N, C]
            trues = np.array(all_labels["cla"][key])                # [N]

            pred_label = preds.argmax(axis=1)
            acc = accuracy_score(trues, pred_label)
            f1_macro = f1_score(trues, pred_label, average="macro")

            metrics = {
                "ACC": round(acc, 4),
                "F1_macro": round(f1_macro, 4),
            }

            # -------- AUC / AUPR（仅在类别齐全时计算） --------
            unique_classes = np.unique(trues).astype(int)
            if len(unique_classes) > 2:
                trues_bin = label_binarize(trues, classes=unique_classes)
                preds_sub = preds[:, unique_classes]
                try:
                    auc_macro = roc_auc_score(trues_bin, preds_sub, average="macro")
                    aupr_macro = average_precision_score(trues_bin, preds_sub, average="macro")
                except ValueError:
                    auc_macro, aupr_macro = float("nan"), float("nan")
                metrics.update({
                    "AUC_macro": round(auc_macro, 4),
                    "AUPR_macro": round(aupr_macro, 4),
                })
            elif len(unique_classes) > 1:
                pos_probs = preds[:, 1]  # 正类概率
                auc = roc_auc_score(trues, pos_probs)
                aupr = average_precision_score(trues, pos_probs)
                metrics.update({
                    "AUC": round(auc, 4),
                    "AUPR": round(aupr, 4),
                })
            else:
                metrics.update({
                    "AUC_macro": float("nan"),
                    "AUPR_macro": float("nan"),
                })

            results["cla"][f"{key}_{task_name}"] = metrics

    return results


def print_cv(model_type, fcv, cv_results_dict, task_weights={}, eva_alpha=0.3):
    """
    打印多任务指标，并根据 task_weights 计算最终 score
    只打印实际存在的 sample_key
    """

    # ====== 1. 生成 header ======
    metrics_keys = []  # (task_type, metric)
    for task_type in ["reg", "cla"]:
        for sk, mdict in cv_results_dict.get(task_type, {}).items():
            for metric in mdict.keys():
                metrics_keys.append((task_type, metric))
    metrics_keys = list(dict.fromkeys(metrics_keys))

    header_items = ["Task", "Model"] + [m for _, m in metrics_keys] + ["Score"]
    header = " ".join(f"{h:<14}" for h in header_items)
    print("-" * len(header), file=fcv)
    print(header, file=fcv)

    # ====== 2. 自动获取 sample_key = 前缀_后缀 ======
    all_keys = list(cv_results_dict.get("reg", {}).keys()) + list(cv_results_dict.get("cla", {}).keys())
    data_types = list(dict.fromkeys(k.rsplit('_', 1)[0] for k in all_keys))
    splits = list(dict.fromkeys(k.rsplit('_', 1)[1] for k in all_keys))

    # ====== 3. 遍历每个 data_type & split，打印行并计算 score ======
    val_score = test_score = test1_score = test2_score = 0.0

    for t_name in data_types:
        for split in splits:
            sample_key = f"{t_name}_{split}"

            # 如果 reg 和 cla 都没有该 key，跳过
            if sample_key not in cv_results_dict.get("reg", {}) \
                    and sample_key not in cv_results_dict.get("cla", {}):
                continue

            line_items = [sample_key, model_type]
            score = 0.0

            # ----- 计算每个 metric -----
            for task_type, metric in metrics_keys:
                value = cv_results_dict.get(task_type, {}).get(sample_key, {}).get(metric, 0.0)
                line_items.append(value)

                # reg task
                if task_type in ["reg", "both"]:
                    if metric in ["Pearson", "Spearman"]:
                        score += value
                    elif metric in ["MAE", "RMSE"]:
                        score -= eva_alpha * value

                # cla task
                elif task_type == "cla":
                    if metric in ["AUC", "AUPR", "AUC_macro", "AUPR_macro", "ACC", "F1_macro"]:
                        score += value

            line_items.append(score)

            # 打印一行
            line = " ".join(
                f"{v:<14.4f}" if isinstance(v, float) else f"{v:<14}"
                for v in line_items
            )
            print(line, file=fcv)

            # 权重累计
            w = task_weights.get(t_name, 1.0)
            if "val" in sample_key:
                val_score += w * score
            elif "te" in sample_key:
                test_score += w * score
            elif "te1" in sample_key:
                test1_score += w * score
            elif "te2" in sample_key:
                test2_score += w * score

    # ====== 4. 最终 score ======
    final_score = 0.5*val_score + 0.55*test_score + 0.55 * test2_score
    return final_score


def evaluate_folds(task_name, df, num_folds, method='pred_mean'):
    results = {}
    y_true = df['true_aff'].values
    preds = [df[f'pred_aff_{i}'].values for i in range(num_folds)]

    if method == 'pred_mean':
        # 每行预测值平均 → 计算指标
        y_pred_mean = np.mean(preds, axis=0)
        mae, rmse, pearson, spearman = reg_metrics(y_true, y_pred_mean)
    elif method == 'eva_mean':
        # 每列计算指标 → 取均值
        metrics = [reg_metrics(y_true, pred) for pred in preds]
        mae, rmse, pearson, spearman = np.mean(metrics, axis=0)
    else:
        raise ValueError("method must be 'pred_mean' or 'eva_mean'")
    results[task_name] = {"MAE": mae, "RMSE": rmse, "Pearson": pearson, "Spearman": spearman, "eval_method": method}
    return results


def cv_evaluate_ML(task_name, all_outputs, all_labels, cv_results, task_type='cla'):
    probs_val = np.array(all_outputs)
    true_val = np.array(all_labels)
    results = {}

    if task_type == 'reg':
        MAE, RMSE, Pearson_corr, Spearman_corr = reg_metrics(true_val, probs_val)
        cv_results[task_name] = {"MAE": MAE, "RMSE": RMSE, "Pearson": Pearson_corr, "Spearman": Spearman_corr}

    else:
        thresholds = np.arange(0.00, 1.01, 0.01)
        predicted_matrix = (probs_val[:, None] > thresholds).astype(int)  # (n_samples, n_thresholds)
        results = np.apply_along_axis(
            lambda preds: cla_metrics(preds, true_val, probs_val),
            axis=0,
            arr=predicted_matrix
        )

        best_idx = np.argmax(results[-1])
        best_cutoff = thresholds[best_idx]
        best_result = results[:, best_idx]

        cv_results[task_name] = {'cutoff': best_cutoff, 'result': best_result.tolist()}

    results[task_type] = cv_results
    return results


def collect_results(outputs_input, labels_input, pairs_input, merge_cv=False, mean_cv=False):
    """
    精简版 collect_results，适配 append_outputs 的输出格式。

    Args:
        outputs_input: list 或单个 dict，格式 {task: {key: list of preds}}
        labels_input: list 或单个 dict，格式 {task: {key: list of labels}}
        pairs_input: list 或单个 dict，样本 pair 信息
        merge_cv: bool, 是否合并所有 fold
        mean_cv: bool, 测试时每个 fold 同样样本，取均值

    Returns:
        merged_out: {task: {key: list of preds}}
        merged_lab: {task: {key: list of labels}}
        merged_pairs: {key: list of pairs}
    """
    # ---- 统一成 folds 列表 ----
    folds_out = outputs_input if isinstance(outputs_input, list) else [outputs_input]
    folds_lab = labels_input if isinstance(labels_input, list) else [labels_input]
    folds_pairs = pairs_input if isinstance(pairs_input, list) else [pairs_input]
    num_folds = len(folds_out)
    tasks = list(folds_out[0].keys())
    keys = list(folds_out[0][tasks[0]].keys())

    # ---- 输出容器 ----
    merged_out = {t: {k: [] for k in keys} for t in tasks}
    merged_lab = {t: {k: [] for k in keys} for t in tasks}
    merged_pairs = {k: [] for k in folds_pairs[0].keys()}

    # =============================
    # A: merge_cv → concat 所有 fold
    # =============================
    if merge_cv:
        for t in tasks:
            for k in keys:
                for f in range(num_folds):
                    merged_out[t][k].extend(folds_out[f][t][k])
                    merged_lab[t][k].extend(folds_lab[f][t][k])
        for f in range(num_folds):
            for k in merged_pairs.keys():
                merged_pairs[k].extend(folds_pairs[f][k])
        return merged_out, merged_lab, merged_pairs

    # =============================
    # B: mean_cv → 测试集，每个样本 fold 上求均值
    # =============================
    if mean_cv:
        for t in tasks:
            for k in keys:
                # 对每个样本在不同 fold 的预测取均值
                preds_per_sample = list(zip(*[folds_out[f][t][k] for f in range(num_folds)]))
                for sample_preds in preds_per_sample:
                    if t == "cla":
                        # tensor [C] → numpy → mean
                        sample_array = torch.stack(sample_preds).float().mean(dim=0)
                        merged_out[t][k].append(sample_array)
                    else:
                        merged_out[t][k].append(float(torch.tensor(sample_preds).mean()))
                # label 直接取第一 fold
                merged_lab[t][k] = folds_lab[0][t][k]

        return merged_out, merged_lab, folds_pairs[0]

    # =============================
    # C: 非 CV → 直接 list
    # =============================
    for t in tasks:
        for k in keys:
            merged_out[t][k] = folds_out[0][t][k]
            merged_lab[t][k] = folds_lab[0][t][k]

    return merged_out, merged_lab, folds_pairs[0]


def get_id_from_best_epochs(fold_best_epoch):
    letters = "abcdefghijklmnopqrstuvwxyz"
    parts = []
    for fold, epoch in sorted(fold_best_epoch.items()):
        parts.append(f"{letters[int(fold)]}{epoch}")
    return "".join(parts)


def save_results_excel(args, tr_outputs, tr_labels, tr_prs, run_id):
    save_path = f"./output/{args.classifier}/{args.model_type}_output_{run_id}.xlsx"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(save_path):
        with pd.ExcelWriter(save_path, engine="openpyxl", mode="w") as writer:
            pd.DataFrame({"init": []}).to_excel(writer, sheet_name="init", index=False)

    writer = pd.ExcelWriter(save_path, engine="openpyxl", mode="a", if_sheet_exists="replace")

    # if os.path.exists(save_path):
    #     writer = pd.ExcelWriter(save_path, engine="openpyxl", mode="a", if_sheet_exists="replace")
    # else:
    #     writer = pd.ExcelWriter(save_path, engine="openpyxl", mode="w")

    with writer:
        for task_type, data_dict in tr_outputs.items():
            for data_type, preds in data_dict.items():
                if preds is None or len(preds) == 0:
                    continue

                labels = np.array(tr_labels[task_type][data_type], dtype=np.float32)
                pairs = tr_prs[data_type]

                # =====================
                # 回归任务
                # =====================
                if task_type != "cla":
                    preds_np = np.array(preds, dtype=np.float32)

                    df = pd.DataFrame({
                        "pair": pairs,
                        "true": labels,
                        "pred": preds_np
                    })
                # =====================
                # 三分类任务
                # =====================
                else:
                    probs = np.array(preds, dtype=np.float32)   # (N, C)
                    # probs = torch.softmax(torch.tensor(preds_np), dim=1).numpy()

                    df = pd.DataFrame({
                        "pair": pairs,
                        "true": labels
                    })

                    for c in range(probs.shape[1]):
                        df[f"prob_c{c}"] = probs[:, c]

                    # 可选：预测类别
                    df["pred_label"] = np.argmax(probs, axis=1)

                # # ========== 分类任务：sigmoid 后取第 3 类 ==========
                # if task_type == "cla":
                #     preds_np = np.array(preds, dtype=np.float32)
                #     probs = torch.sigmoid(torch.tensor(preds_np))[:, 2].numpy()
                # # ========== 回归任务：统一 numpy ==========
                # else:
                #     probs = np.array(preds, dtype=np.float32)
                #
                # df = pd.DataFrame({
                #     "pair": pairs,
                #     "true": labels,
                #     "pred": probs
                # })

                df.to_excel(writer, sheet_name=f"{task_type}_{data_type}_{args.dataset}",
                            index=False, float_format="%.5f")

    print(f"Results saved to {save_path}")


def save_output(pc, pred_result, pc_info, output_file):
    """
    整合抗体-抗原接触分数信息并保存到txt文件。

    参数：
    pred_result (np.ndarray): M*N 形状的抗体-抗原接触分数矩阵
    ab_info (np.ndarray): 形状 (M, 3)，包含抗体的 (res_type, chain, res_id)
    ag_info (np.ndarray): 形状 (N, 3)，包含抗原的 (res_type, chain, res_id)
    output_file (str): 结果输出文件名
    """

    results = []
    index = 1
    ab_info, ag_info = pc_info

    for i in range(pred_result.shape[0]):
        for j in range(pred_result.shape[1]):
            score = pred_result[i, j]

            res_num1 = f"{ab_info[i, 2]}:{ab_info[i, 1]}"  # res_id:chain
            res_name1 = ab_info[i, 0]  # res_type

            res_num2 = f"{ag_info[j, 2]}:{ag_info[j, 1]}"
            res_name2 = ag_info[j, 0]

            results.append([index, res_num1, res_name1, res_num2, res_name2, f"{score:.4f}"])
            index += 1

    df = pd.DataFrame(results, columns=["Number", "ResNum1", "ResName1", "ResNum2", "ResName2", "Predicted_Score"])

    df.to_csv(output_file, sep="\t", index=False)
