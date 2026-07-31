import argparse


def arg_parse():
    parser = argparse.ArgumentParser(
        description="UniBA full training-reproduction workspace: preprocessing-compatible training and testing settings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--seed", type=int, default=2025, help="random seed")
    parser.add_argument("--gpuid", type=int, default=0, help="GPU id used for training / inference")
    parser.add_argument("--pdb", type=str, default="./data/pdb_files", help="raw structure directory")
    parser.add_argument("--data_path", type=str, default="./data/affinity_data", help="prepared data directory")
    parser.add_argument("--feat_path", type=str, default="./feature", help="prepared feature directory")
    parser.add_argument("--label_path", type=str, default="./data/", help="label root directory")

    parser.add_argument("--ghecom", type=str, default="", help="optional path to ghecom")
    parser.add_argument("--dssp", type=str, default="", help="optional path to mkdssp")
    parser.add_argument("--psaia", type=str, default="./software/psaia", help="optional path to PSAIA")
    parser.add_argument("--esm_path", type=str, default="", help="optional path to ESM model cache")
    parser.add_argument("--tmalign", type=str, default="", help="optional path to TMalign")
    parser.add_argument("--nwalign", type=str, default="", help="optional path to NWalign")
    parser.add_argument("--tm_library", type=str, default="./data", help="optional TM library data directory")

    parser.add_argument("--modules_path", type=str, default="./ablation", help="directory containing checkpoints")
    parser.add_argument("--output_path", type=str, default="./output", help="output directory")
    parser.add_argument("--cv", type=int, default=1, help="number of folds to use")
    parser.add_argument("--num_epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=6e-5, help="base learning rate")
    parser.add_argument("--lr_seq", type=float, default=1e-5, help="sequence module learning rate")
    parser.add_argument("--lr_str", type=float, default=2e-5, help="structure module learning rate")
    parser.add_argument("--lr_cg", type=float, default=6e-4, help="coarse-grained module learning rate")
    parser.add_argument("--lr_res", type=float, default=6e-5, help="residue module learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--model_type", type=str, default="UniBA", help="model name")
    parser.add_argument("--split_level", type=str, default="cv", help="split setting")
    parser.add_argument("--task_mode", type=str, default="reg", help="task mode")
    parser.add_argument("--classifier", type=str, default="UniBA_wo_res", help="checkpoint subdirectory")
    parser.add_argument("--dataset", type=str, default="test", help="dataset split name for inference")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--max-train-samples", type=int, default=0, help="optional cap for training samples per fold")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="optional cap for validation / test samples")
    parser.add_argument(
        "--checkpoint-json",
        type=str,
        default="configs/UniBA_fold_best_epoch.json",
        help="JSON file used to store the best epoch for each fold",
    )
    parser.add_argument(
        "--init-from-release",
        action="store_true",
        help="initialize each fold from the copied release checkpoint before training",
    )
    parser.add_argument(
        "--save-checkpoints",
        dest="save_checkpoints",
        action="store_true",
        help="save best checkpoints during training",
    )
    parser.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false", help=argparse.SUPPRESS)

    parser.set_defaults(train=False, test=True, save_checkpoints=True)
    parser.add_argument("--train", dest="train", action="store_true", help="enable training")
    parser.add_argument("--no-train", dest="train", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--test", dest="test", action="store_true", help="enable inference")
    parser.add_argument("--no-test", dest="test", action="store_false", help="disable inference")

    return parser.parse_args()
