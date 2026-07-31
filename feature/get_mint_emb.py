import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

local_mint_root = os.path.join(os.path.dirname(__file__), "mint")
if os.path.isdir(local_mint_root) and local_mint_root not in sys.path:
    sys.path.insert(0, local_mint_root)

from mint.helpers.extract import CSVDataset, CollateFn, MINTWrapper, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sequence-level MINT embeddings for prepared protein pairs."
    )
    parser.add_argument("--config", required=True, help="Path to the MINT config JSON")
    parser.add_argument("--checkpoint", required=True, help="Path to the MINT checkpoint")
    parser.add_argument("--csv", required=True, help="CSV file containing paired sequences")
    parser.add_argument("--pair-list", required=True, help="Text file listing pair ids in CSV order")
    parser.add_argument("--output", required=True, help="Output .pt file")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length for CollateFn")
    parser.add_argument(
        "--no-sep-chains",
        dest="sep_chains",
        action="store_false",
        help="Disable separated-chain mode in MINTWrapper",
    )
    parser.set_defaults(sep_chains=True)
    return parser.parse_args()


def main():
    args = parse_args()
    csv_file = Path(args.csv)
    pair_list_file = Path(args.pair_list)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    dataset = CSVDataset(str(csv_file), "Protein_Sequence_1", "Protein_Sequence_2")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=CollateFn(args.max_length),
        shuffle=False,
    )

    wrapper = MINTWrapper(cfg, args.checkpoint, sep_chains=args.sep_chains, device=args.device)
    pair_names = np.loadtxt(pair_list_file, dtype=str)
    if pair_names.ndim == 0:
        pair_names = np.array([str(pair_names)])
    _ = pd.read_csv(csv_file)

    all_embeddings = {}
    for batch_idx, (chains, chain_ids) in enumerate(loader):
        chains = chains.to(args.device)
        chain_ids = chain_ids.to(args.device)
        emb = wrapper(chains, chain_ids).cpu()

        for i in range(emb.shape[0]):
            global_idx = batch_idx * loader.batch_size + i
            if global_idx >= len(pair_names):
                break
            pair_name = str(pair_names[global_idx])
            print(global_idx, pair_name)
            all_embeddings[pair_name] = emb[i]

    torch.save(all_embeddings, output_path)
    print(f"[Done] Saved {len(all_embeddings)} embeddings to {output_path}")


if __name__ == "__main__":
    main()
