import argparse
import pathlib
import string
import torch
from esm import Alphabet, pretrained, MSATransformer
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import itertools

#Clean a string, converting lowercase letters, removing periods and asterisks 

def remove_insertions(sequence: str) -> str: 
    #Create dict where each key is lowercase letter and value is None 
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    #Create a mapping table 
    translation = str.maketrans(deletekeys)
    #Removes any char in seq with a value of None
    return sequence.translate(translation)


def create_parser():
    parser = argparse.ArgumentParser(description="Label a deep mutational scan with predictions from an ensemble of ESM-2 models.")
    parser.add_argument("--model-location", type=str, help="PyTorch model file OR name of pretrained model to download", nargs="+")
    parser.add_argument("--sequence", type=str, help="Base sequence to which mutations were applied")
    parser.add_argument("--dms-input", type=pathlib.Path, help="CSV file containing the deep mutational scan")
    parser.add_argument("--mutation-col", type=str, default="mutant", help="column in the deep mutational scan labeling the mutation as 'AiB', where A is WT, i is pos and B is mutant AA")
    parser.add_argument("--dms-output", type=pathlib.Path, help="Output file containing the deep mutational scan along with predictions")
    parser.add_argument("--offset-idx", type=int, default=0, help="Offset of the mutation positions in `--mutation-col`")
    parser.add_argument("--scoring-strategy", type=str, default="wt-marginals", choices=["wt-marginals", "pseudo-ppl", "masked-marginals"], help="")
    parser.add_argument("--msa-path", type=pathlib.Path, help="path to MSA in a3m format (required for MSA Transformer)")
    parser.add_argument("--msa-samples", type=int, default=400, help="number of sequences to select from the start of the MSA")
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser

def label_row(row, sequence, token_probs, alphabet, offset_idx):
    """"
    Calculates a score for a single mutation based on the difference in log probabilities between the mutant and wild type amino acids at a specific position in the protein sequence. row is AiB indicator from DMS CSV. token_probs contains log propbs for each AA at each mutation. alphabet is encoding of AA and offset idx adjusts mutation pos index if there's an offset between sequence indexing & DMS data """"
    #wt is first char of row, idx is middle part of row, and mutant is last parrt of row
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]i 
    #Validate WT AA at specified position 
    assert sequence[idx] == wt, "Listed WT does not match the provided sequence"
    #Encode Amino Acids 
    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
    #Calciulate the score, difference in log probabilities
    score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
    #Convert tensor to a scalar 
    return score.item()

def compute_pppl(row, sequence, model, alphabet, offset_idx):
    """"Calculates the pseudo perplexity (PPPL) for a specfici mutation, which is a metric for how well a model predicts the probability of a sequence""""
    #Extract WT, position  & mutant AAs 
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    #Encure WT AA in seq matches what was specified 
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
    #Replace WT AA at pos idx w mutant amino acid mt 
    sequence = sequence[:idx] + mt + sequence[(idx + 1):]
    #Prepare data for model input 
    data = [("protein1", sequence)]
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    log_probs = []
    #Iterate over each position in the sequence except for special tokens 
    for i in range(1, len(sequence)-1): 
        #Copy of original tokens w AA at pos i replaced by masked token
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            #Passes masked toknens through the model & applies log softmax to get log probabilities
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
            #Collects log probabilities for each position
            log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())
    #Aggregates log probabilities across all positions, representing pseudoperplexity for mutated seq
    return sum(log_probs) 

def main(args): 
    df = pd.read_csv(args.dms_input)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.nogpu else "cpu")
    for model_location in args.model_location:
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
        model = model.to(device)
        model.eval()
        batch_converter = alphabet.get_batch_converter()
        if isinstance(model, MSATransformer):
            #Reads MSA file and selects the first msa_samples sequences
            data = [read_msa(args.msa_path, args.msa_samples)]
            assert args.scoring_strategy == "masked-marginals", "MSA Transformer only supports masked marginal strategy"
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            #Compute log probs for each AA at every pos in MSA by masking one token at a time 
            all_token_probs = []
            for i in tqdm(range(batch_tokens.size(2))):
                batch_tokens_masked = batch_tokens.clone()
                batch_tokens_masked[0, 0, i] = alphabet.mask_idx
                with torch.no_grad():
                    token_probs = torch.log_softmax(model(batch_tokens_masked)["logits"], dim=-1)
                all_token_probs.append(token_probs[:, 0, i])
            #Tensor containing log probabilities for each position 
            token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
            #Applies label_row function to each row in DMS df
            df[model_location] = df.apply(
                lambda row: label_row(row[args.mutation_col], args.sequence, token_probs, alphabet, args.offset_idx),
                axis=1,
            )
    else:
        data = [("protein1", args.sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        if args.scoring_strategy == "wt-marginals":
            with torch.no_grad():
                token_probs = torch.log_softmax(model(batch_tokens)["logits"], dim=-1)
            df[model_location] = df.apply(
                    lambda row: label_row(row[args.mutation_col], args.sequence, token_probs, alphabet, args.offset_idx),
                    axis=1,
                )
        elif args.scoring_strategy == "masked-marginals":
                all_token_probs = []
                for i in tqdm(range(batch_tokens.size(1))):
                    batch_tokens_masked = batch_tokens.clone()
                    batch_tokens_masked[0, i] = alphabet.mask_idx
                    with torch.no_grad():
                        token_probs = torch.log_softmax(model(batch_tokens_masked)["logits"], dim=-1)
                    all_token_probs.append(token_probs[:, i])
                token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
                df[model_location] = df.apply(
                    lambda row: label_row(row[args.mutation_col], args.sequence, token_probs, alphabet, args.offset_idx),
                    axis=1,
                )
        elif args.scoring_strategy == "pseudo-ppl":
                tqdm.pandas()
                df[model_location] = df.progress_apply(
                    lambda row: compute_pppl(row[args.mutation_col], args.sequence, model, alphabet, args.offset_idx),
                    axis=1,
                )
        df.to_csv(args.dms_output)
if __name__ == "__main__":
    #Set up CLI 
    parser = create_parser()
    #Parse args by user when running script
    args = parser.parse_args()
    main(args)


            



    


