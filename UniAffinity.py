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


def _fold_score(eval_result, task_mode):
    score = 0.0
    for metrics in eval_result.get("reg", {}).values():
        score += metrics.get("Pearson", 0.0)
        score += metrics.get("Spearman", 0.0)
        score -= 0.3 * metrics.get("RMSE", 0.0)
        score -= 0.3 * metrics.get("MAE", 0.0)

    if task_mode in ["cla", "both"]:
        for metrics in eval_result.get("cla", {}).values():
            score += metrics.get("AUC", 0.0)
            score += metrics.get("AUPR", 0.0)
            score += metrics.get("ACC", 0.0)
            score += metrics.get("F1_macro", 0.0)

    return score


def _write_train_header(log_path, args):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fcv:
        print(
            f"model={args.model_type}\n"
            f"classifier={args.classifier}\n"
            f"train={args.train}, test={args.test}\n"
            f"cv={args.cv}, batch_size={args.batch_size}, num_epochs={args.num_epochs}\n"
            f"lr={args.lr}, lr_seq={args.lr_seq}, lr_str={args.lr_str}, "
            f"lr_cg={args.lr_cg}, lr_res={args.lr_res}, wd={args.wd}\n"
            f"task_mode={args.task_mode}, split_level={args.split_level}\n"
            f"init_from_release={args.init_from_release}, save_checkpoints={args.save_checkpoints}\n",
            file=fcv,
        )


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
    fold_best_epoch = {}
    released_best_epoch = {}
    if args.init_from_release and os.path.exists(args.checkpoint_json):
        with open(args.checkpoint_json, "r") as fe:
            released_best_epoch = json.load(fe)

    for fold in range(0, args.cv):
        tr_pr_list, val_pr_list = get_pr_list(args, fold=fold)
        if args.max_train_samples > 0:
            tr_pr_list = tr_pr_list[:args.max_train_samples]
        if args.max_eval_samples > 0:
            val_pr_list = val_pr_list[:args.max_eval_samples]

        model = UniAffinityNet(seq_emb_dim=1280, seq_hd_dim=512, res_node_dim=1813,
                               res_edge_dim=20, res_hd_dims=[512],
                               cg_node_dim=35, cg_edge_dim=51, cg_hd_dims=[256] * 2,
                               cg_gh_builder=cg_graph_model, task_mode=args.task_mode,
                               ensemble_mode="repr_residual").to(device)
        if args.init_from_release:
            release_epoch = released_best_epoch[str(fold)]
            checkpoint_path = f'{args.modules_path}/{args.classifier}/{args.model_type}_{fold}_{release_epoch}.pth'
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        best_score = -np.inf
        best_state_dict = copy.deepcopy(model.state_dict())
        best_val_output, best_val_label, best_val_pair = None, None, None
        best_epoch = 0
        no_improve = 0
        patience = max(1, args.patience)
        log_path = f'{args.modules_path}/{args.classifier}/{args.model_type}_{fold}.txt'
        _write_train_header(log_path, args)

        for epoch in range(args.num_epochs):
            model.train()
            tr_label, tr_output, tr_pair, tr_loss, _ = loop(
                model, tr_pr_list, mint_emb, data_dict, optimizer, fold=fold)

            model.eval()
            results_all = {'reg': {}, 'cla': {}}
            with torch.no_grad():
                val_label, val_output, val_pair, val_loss, _ = loop(
                    model, val_pr_list, mint_emb, data_dict, fold=fold)
                val_eval = epoch_evaluate("val", val_label, val_output, args.task_mode)
                merge_result(results_all, val_eval)

            cur_score = _fold_score(results_all, args.task_mode)
            with open(log_path, "a", encoding="utf-8") as fcv:
                print(
                    f"\nEpoch:{epoch + 1}, train_loss:{float(tr_loss):.4f}, val_loss:{float(val_loss):.4f}, score:{cur_score:.4f}",
                    file=fcv,
                )
                task_weights = {'wt': 1.0, 'mut': 1.0, 'all': 1.0}
                print_cv(args.model_type, fcv, results_all, task_weights, eva_alpha=0.3)

            if cur_score > best_score:
                best_score = cur_score
                best_state_dict = copy.deepcopy(model.state_dict())
                best_val_output = copy.deepcopy(val_output)
                best_val_label = copy.deepcopy(val_label)
                best_val_pair = copy.deepcopy(val_pair)
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        if args.save_checkpoints:
            checkpoint_path = f'{args.modules_path}/{args.classifier}/{args.model_type}_{fold}_{best_epoch}.pth'
            torch.save(best_state_dict, checkpoint_path)

        fold_best_epoch[str(fold)] = best_epoch
        cv_outputs.append(best_val_output)
        cv_labels.append(best_val_label)
        cv_pairs.append(best_val_pair)

    tr_outputs, tr_labels, tr_pairs = collect_results(cv_outputs, cv_labels, cv_pairs, merge_cv=True)

    with open(args.checkpoint_json, "w") as fe:
        json.dump(fold_best_epoch, fe, indent=4)
    model_id = get_id_from_best_epochs(fold_best_epoch)

    return tr_outputs, tr_labels, tr_pairs, model_id


def model_test(args, data_dict, mint_emb, cg_graph_model, device):
    # fold_best_epoch = {0: 15, 1: 28, 2: 26, 3: 24, 4: 35}
    # with open(f"{args.model_type}_fold_best_epoch.json", "w") as fe:
    #     json.dump(fold_best_epoch, fe, indent=4)
    with open(args.checkpoint_json, "r") as fe:
        fold_best_epoch = json.load(fe)

    # te_wt_pr_list = get_pr_list(args, args.dataset)
    # te_mut_pr_list = get_mut_pr_list(args, args.dataset)
    # pr_list = np.concatenate([te_wt_pr_list, te_mut_pr_list])
    pr_list = get_pr_list(args, args.dataset)
    if args.max_eval_samples > 0:
        pr_list = pr_list[:args.max_eval_samples]
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
    seed = args.seed
    set_random_seed(seed)
    cg_graph_model = CG22_GraphConstruction()
    mint_emb = load_all_seq_embeddings(plm='mint')

    with open(f"{args.data_path}/all_data_with_multi_labels.json", "r") as f:
        data_dict = json.load(f)
    # interface_feat_file = "/home/yyShen/NAcontact/feature/interface_dict.pkl"
    # if os.path.exists(interface_feat_file):
    #     with open(interface_feat_file, "rb") as f:
    #         interface_data = pickle.load(f)
    # else:
    #     raise FileNotFoundError("interface_dict.pkl not found")
    # interface_dict = interface_data["interface_dict"]

    if args.train:
        original_dataset = args.dataset
        args.dataset = "val"
        tr_prob, tr_label, tr_pair, model_id = model_train(
            args, data_dict, mint_emb, cg_graph_model, device)
        save_results_excel(args, tr_prob, tr_label, tr_pair, model_id)
        args.dataset = original_dataset

    if args.test:
        # stats = interface_data["stats"]
        te_prob, te_label, te_pair, model_id = model_test(
            args, data_dict, mint_emb, cg_graph_model, device)
        save_results_excel(args, te_prob, te_label, te_pair, model_id)
