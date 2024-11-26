from transformers import AutoTokenizer, EsmForMaskedLM
import torch
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display

def generate_heatmap(protein_sequence, start_pos=1, end_pos=None):
        model_name = "facebook/esm2_t6_8M_UR50D"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmForMaskedLM.from_pretrained(model_name)
        input_ids = tokenizer.encode(protein_sequence, return_tensors ="pt")
        #Computes actual length of protein sequence
        sequence_length = input_ids.shape[1] - 2

        if end_pos is None:
                end_pos = sequence_length
        amino_acids = list("ASDEFGHIKLMNPQRSTVWY")
        heatmap = np.zeros((20, end_pos - start_pos + 1))

        for position in range(start_pos, end_pos +1):
                #Creates a copy of input token IDs
                masked_input_ids = input_ids.clone()
                #Replaces token at current pos with mask token
                masked_input_ids[0, position] = tokenizer.mask_token_id
        #Get raw prediction scores for each possible AA at masked position  
                with torch.no_grad(): 
                    logits = model(masked_input_ids).logits
                #Convert logits into probabilities 
                probabilities = torch.nn.functional.softmax(logits[0, position], dim=0)
                log_probabilities = torch.log(probabilities) 
                wt_residue = input_ids[0, position].item()
                log_prob_wt = log_probabilities[wt_residue].item()
        
        #Calulcate LLR for each variant
                for i, amino_acid in enumerate(amino_acids): 
                    log_prob_mt = log_probabilities[tokenizer.convert_tokens_to_ids(amino_acid)].item()
                    heatmap[i, position-start_pos] = log_prob_mt - log_prob_wt
        plt.figure(figsize=(15, 5))
        plt.imshow(heatmap, cmap="viridis", aspect="auto")
        plt.xticks(range(end_pos - start_pos + 1), list(protein_sequence[start_pos-1:end_pos]))
        plt.yticks(range(20), amino_acids)
        plt.xlabel("Position in Protein Sequence")
        plt.ylabel("Amino Acid Mutations")
        plt.title("Predicted Effects of Mutations on Protein Sequence (LLR)")
        plt.colorbar(label="Log Likelihood Ratio (LLR)")
        plt.show()

def interactive_heatmap(protein_sequence): 
    #Create 2 int sliders to select start and end positions in protein seq 
    start_slider = widgets.IntSlider(value=1, min=1, max=len(protein_sequence), step=1, description='Start:')
    end_slider = widgets.IntSlider(value=len(protein_sequence), min=1, max=len(protein_sequence), step=1, description='End:')
    ui = widgets.HBox([start_slider, end_slider]) 
    def update_heatmap(start, end): 
        if start <= end: 
            generate_heatmap(protein_sequence, start, end)
    out = widgets.interactive_output(update_heatmap, {'start':start_slider, 'end':end_slider})
    display(ui, out)

protein_sequence = "MAPLRKTYVLKLYVAGNTPNSVRALKTLNNILEKEFKGVYALKVIDVLKNPQLAEEDKILATPTLAKVLPPPVRRIIGDLSNREKVLIGLDLLYEEIGDQAEDDLGLE"
interactive_heatmap(protein_sequence)


