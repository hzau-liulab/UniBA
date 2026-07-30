import sys
import os

# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
unique_paths = []
for path in sys.path:
    if path not in unique_paths:
        unique_paths.append(path)
sys.path = unique_paths
from utils.model_utils import *
from arg_parse import *
from utils.cg_graphconstruct import *
from model.UniAffinityNet import UniAffinityNet
import copy


def loop(model, all_pr_list, mint_emb, data_dict=None, optimizer=None,
         task_mode="reg", bce_weight=1, fold=None
         ):

    types = ["wt"]
    sample_weight = {"ppi": 1.0, "aai": 1.0, "tcr-pmhc": 1.0}
    out_dict = None
    all_outputs = init_nested_dict(task_mode, types)
    all_labels = init_nested_dict(task_mode, types)
    all_prs = {tp: [] for tp in types}
    all_out_dicts = defaultdict(dict) if optimizer is None else None

    total_loss, total_steps = 0.0, 0

    for k in range(0, len(all_pr_list), args.batch_size):
        batch_pr = all_pr_list[k:k + args.batch_size]
        batch_cla_loss, batch_reg_loss = [], []

        for pair in batch_pr:
            # pair = "1g0v_A_B"
            # print(pair)
            label, wt_entry, iface = build_label_dict(pair, data_dict, device)
            seq_data, key = get_seq_embedding(pair, wt_entry, mint_emb, device)
            res_graph, _ = get_str_graph(args, pair, wt_entry, 'res')
            cg_graph, _ = get_str_graph(args, pair, wt_entry, 'cg_intra')

            w = sample_weight.get(wt_entry["type"], 1.0)
            if task_mode in ["reg", "both"]:
                out_dict = model(seq_data, res_graph, cg_graph)
                # pKd_pred = out_dict["pKd"]
                # reg_loss = mse_scaled(pKd_pred, label["pKd"]) * w
                # print(pair, label["pKd"], out_dict["pKd"], out_dict["y_str"], out_dict["y_seq"])
                reg_loss = compute_loss(out_dict, label)
                batch_reg_loss.append(reg_loss)
                append_outputs(all_outputs, all_labels, 'wt', out_dict["pKd"], label["pKd"], "reg")

            if task_mode in ["cla", "both"]:  # and epoch is not None and epoch >= stage_bce_epoch:
                out_dict = model(seq_data, res_graph, cg_graph, iface, wt_entry["type"])
                aff_logits = out_dict["cla"]
                # cla_loss = F.binary_cross_entropy_with_logits(cla_pred, label_true)
                cla_loss = F.cross_entropy(aff_logits, label['label_aff_cls'])
                batch_cla_loss.append(cla_loss)
                append_outputs(all_outputs, all_labels, key, aff_logits, label['label_aff_cls'], "cla")

            if all_out_dicts is not None:
                all_out_dicts[(fold, pair)] = {
                    "pred": out_dict["pKd"].detach().cpu().item(),
                    "y_seq": out_dict["y_seq"].detach().cpu().item(),
                    "y_str": out_dict["y_str"].detach().cpu().item(),
                    "seq_gate": out_dict["seq_gate"].detach().cpu().numpy(),
                    "pair_attn": out_dict["pair_attn"],
                    "true": label["pKd"].detach().cpu().item()
                }

                # all_out_dicts[(fold, pair)] = out_dict

            all_prs['wt'].append(pair)
            # all_prs["all"].append(pair)
            # append_outputs(all_outputs, all_labels, "all", cla_pred, label_true, task_mode)

        loss = torch.zeros((), device=device)
        if batch_reg_loss:
            loss += torch.stack(batch_reg_loss).mean()
        if batch_cla_loss:
            loss += torch.stack(batch_cla_loss).mean() * bce_weight

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_steps += 1

    all_loss = round(total_loss / total_steps, 4)
    return all_labels, all_outputs, all_prs, all_loss, all_out_dicts


def model_train(args, data_dict, mint_emb, cg_graph_model, device):
    cv_outputs, cv_labels, cv_pairs = [], [], []
    cv_val_out_dicts = []
    fold_best_epoch = {}
    with open(f"{args.model_type}_fold_best_epoch.json", "r") as fe:
        fold_best_epoch = json.load(fe)

    for fold in range(0, args.cv):
        # with open(f'{args.modules_path}/{args.classifier}/{args.model_type}_{fold}.txt', 'w') as fcv:
        #     print(f'epoch:{args.num_epochs}, batch_size:{args.batch_size}, lr:{args.lr}, lr_seq:{args.lr_seq}, '
        #           f'lr_res:{args.lr_res}, wd:{args.wd}, use_type=False, ensemble_mode=repr_residual, '
        #           f'seq_emb_dim=1280, seq_hd_dim=512, hd_dim=256, '
        #           f'res_node_dim=1813, res_hd:[512], res_edge_dim=20, GraphAtt_1, '
        #           f'cg_node_dim=35, cg_edge_dim=53, num_relation=7, cg_hd=[256]*2, num_angle_bin=6, cg_pool_bead, '
        #           f'{args.model_type}, task_mode={args.task_mode}, {args.split_level}, wt', file=fcv)

        tr_pr_list, val_pr_list = get_pr_list(args, fold=fold)
        te_pr_list = get_pr_list(args, 'test')

        model = UniAffinityNet(seq_emb_dim=1280, seq_hd_dim=512, res_node_dim=1813,
                               res_edge_dim=20, res_hd_dims=[512],
                               cg_node_dim=35, cg_edge_dim=51, cg_hd_dims=[256] * 2,
                               cg_gh_builder=cg_graph_model, task_mode=args.task_mode,
                               ensemble_mode="repr_residual").to(device)
        best_epoch = fold_best_epoch[str(fold)]
        checkpoint_path = f'{args.modules_path}/{args.classifier}/{args.model_type}_{fold}_{best_epoch}.pth'
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # model, key_dict = load_pretrained_module(model, fold, pre_fold_best_epoch, device=device)
        # model.set_requires_grad(key_dict["backbone"], False)
        # model.set_requires_grad(key_dict["fusion"], True)

        # optimizer = torch.optim.Adam(
        #     [
        #         # {"params": collect_params_by_keys(model, key_dict["seq"]), "lr": args.lr_seq},
        #         # {"params": collect_params_by_keys(model, key_dict["str"]), "lr": args.lr_cg},
        #         {"params": collect_params_by_keys(model, key_dict["fusion"]), "lr": args.lr},
        #     ],
        #     weight_decay=args.wd
        # )

        best_score = -np.inf
        best_state_dict, best_val_output, best_val_out_dicts = None, None, None
        best_epoch, no_improve = 0, 0
        patience = 3

        # for epoch in range(args.num_epochs):
        #     model.train()
        #     tr_label, tr_output, tr_pair, tr_loss, _ = loop(
        #         model, tr_pr_list, mint_emb, data_dict, optimizer, fold=fold)

        model.eval()
        results_all = {'reg': {}, 'cla': {}}
        with torch.no_grad():
            val_label, val_output, val_pair, val_loss, val_out_dicts = loop(
                model, val_pr_list, mint_emb, data_dict, fold=fold)
            val_eval = epoch_evaluate("val", val_label, val_output, args.task_mode)
            merge_result(results_all, val_eval)

        #         te_label, te_output, te_pair, _, _ = loop(
        #             model, te_pr_list, mint_emb, data_dict, fold=fold)
        #         te_eval = epoch_evaluate("te", te_label, te_output, args.task_mode)
        #         merge_result(results_all, te_eval)
        #
        #     with open(f'{args.modules_path}/{args.classifier}/{args.model_type}_{fold}.txt', 'a') as fcv:
        #         print(f"\nEpoch:{epoch + 1}, tr_loss:{np.mean(tr_loss)}, val_loss:{np.mean(val_loss)}", file=fcv)
        #         task_weights = {'wt': 1.0, 'mut': 1.0, 'all': 1.0}
        #         cur_score = print_cv(args.model_type, fcv, results_all, task_weights, eva_alpha=0.4)
        #
        #     if cur_score > best_score:
        #         best_score = cur_score
        #         best_state_dict = copy.deepcopy(model.state_dict())
        #         best_val_output = val_output
        #         best_val_out_dicts = val_out_dicts
        #         best_epoch = epoch + 1
        #         no_improve = 0
        #     else:
        #         no_improve += 1
        #
        #     if no_improve >= patience:
        #         break
        #
        # fold_best_epoch[fold] = best_epoch
        # torch.save(best_state_dict,
        #            f'{args.modules_path}/{args.classifier}/{args.model_type}_{fold}_{best_epoch}.pth')

        cv_outputs.append(val_output)
        cv_labels.append(val_label)
        cv_pairs.append(val_pair)
        cv_val_out_dicts.append(best_val_out_dicts)

        # # 关键修改：保存 fold 信息
        # fold_val_dict = {}
        # for pair, out_dict in best_val_out_dicts.items():
        #     fold_val_dict[(fold, pair)] = out_dict
        # cv_val_out_dicts.append(fold_val_dict)

    tr_outputs, tr_labels, tr_pairs = collect_results(cv_outputs, cv_labels, cv_pairs, merge_cv=True)

    # merged_val_out_dicts = {}
    # for d in cv_val_out_dicts:
    #     merged_val_out_dicts.update(d)

    # with open(f"{args.model_type}_fold_best_epoch.json", "w") as fe:
    #     json.dump(fold_best_epoch, fe, indent=4)
    model_id = get_id_from_best_epochs(fold_best_epoch)

    # out_path = f"./analysis_graph/out_info/{args.model_type}_{args.dataset}_{model_id}.pt"
    # os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # torch.save(cv_val_out_dicts, out_path)

    return tr_outputs, tr_labels, tr_pairs, model_id


def model_test(args, data_dict, mint_emb, cg_graph_model, device):
    # fold_best_epoch = {0: 15, 1: 28, 2: 26, 3: 24, 4: 35}
    # with open(f"{args.model_type}_fold_best_epoch.json", "w") as fe:
    #     json.dump(fold_best_epoch, fe, indent=4)
    with open(f"{args.model_type}_fold_best_epoch.json", "r") as fe:
        fold_best_epoch = json.load(fe)

    # te_wt_pr_list = get_pr_list(args, args.dataset)
    # te_mut_pr_list = get_mut_pr_list(args, args.dataset)
    # pr_list = np.concatenate([te_wt_pr_list, te_mut_pr_list])
    pr_list = get_pr_list(args, args.dataset)
    # pr_list = ['1n8z_BA_C', '1s78_DC_A']
    te_outputs, te_labels, te_pairs = [], [], []
    fold_out_dict = []
    for fold in range(0, args.cv):
        model = UniAffinityNet(seq_emb_dim=1280, seq_hd_dim=512, res_node_dim=1813,
                               res_edge_dim=20, res_hd_dims=[512],
                               cg_node_dim=35, cg_edge_dim=51, cg_hd_dims=[256] * 2,
                               cg_gh_builder=cg_graph_model, task_mode=args.task_mode,
                               ensemble_mode="repr_residual").to(device)
        best_epoch = fold_best_epoch[str(fold)]
        print(fold, best_epoch)
        checkpoint_path = f'{args.modules_path}/{args.classifier}/{args.model_type}_{fold}_{best_epoch}.pth'
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        model.eval()
        with torch.no_grad():
            pr_label, pr_probs, pr_pair, _, out_dicts = loop(
                model, pr_list, mint_emb, data_dict, fold=fold)

        te_outputs.append(pr_probs)
        te_labels.append(pr_label)
        te_pairs.append(pr_pair)
        fold_out_dict.append(out_dicts)

    merged_te_out_dicts = {}
    for d in fold_out_dict:
        merged_te_out_dicts.update(d)

    te_outputs, te_labels, te_pairs = collect_results(te_outputs, te_labels, te_pairs, merge_cv=False, mean_cv=True)
    model_id = get_id_from_best_epochs(fold_best_epoch)

    # out_path = f"./analysis_graph/{args.classifier}/{args.model_type}_case_{model_id}.pt"
    out_path = f"./analysis_graph/analysis_test/{args.classifier}/{args.model_type}_{args.dataset}_{model_id}.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(merged_te_out_dicts, out_path)

    return te_outputs, te_labels, te_pairs, model_id


if __name__ == '__main__':
    args = arg_parse()
    device = torch.device("cuda:%s" % args.gpuid if torch.cuda.is_available() else "cpu")
    seed = 2025
    set_random_seed(seed)
    cg_graph_model = CG22_GraphConstruction()
    mint_emb = load_all_seq_embeddings(plm='mint')

    with open(f"{args.data_path}/all_data_with_multi_labels.json", "r") as f:
        data_dict = json.load(f)
    # with open(f"{args.data_path}/c-met_data_dict.json", "r") as f:
    #     data_dict = json.load(f)

    # interface_feat_file = "/home/yyShen/NAcontact/feature/interface_dict.pkl"
    # if os.path.exists(interface_feat_file):
    #     with open(interface_feat_file, "rb") as f:
    #         interface_data = pickle.load(f)
    # else:
    #     raise FileNotFoundError("interface_dict.pkl not found")
    # interface_dict = interface_data["interface_dict"]

    with open("mint_fold_best_epoch.json", "r") as fs:
        seq_fold_best_epoch = json.load(fs)
    with open("str_fold_best_epoch.json", "r") as fc:
        str_fold_best_epoch = json.load(fc)
    with open("res_fold_best_epoch.json", "r") as fe:
        res_fold_best_epoch = json.load(fe)
    with open("cg_fold_best_epoch.json", "r") as fg:
        cg_fold_best_epoch = json.load(fg)
    # with open("UniBA_moe_fold_best_epoch.json", "r") as fo:
    #     moe_fold_best_epoch = json.load(fo)
    # with open("UniBA_repr_fold_best_epoch.json", "r") as fr:
    #     repr_fold_best_epoch = json.load(fr)
    pre_fold_best_epoch = {'seq': seq_fold_best_epoch, 'str': str_fold_best_epoch,
                           'res': res_fold_best_epoch, 'cg': cg_fold_best_epoch}

    if args.train:
        tr_prob, tr_label, tr_pair, model_id = model_train(
            args, data_dict, mint_emb, cg_graph_model, device)
        save_results_excel(args, tr_prob, tr_label, tr_pair, model_id)

    if args.test:
        # stats = interface_data["stats"]
        te_prob, te_label, te_pair, model_id = model_test(
            args, data_dict, mint_emb, cg_graph_model, device)
        save_results_excel(args, te_prob, te_label, te_pair, model_id)
