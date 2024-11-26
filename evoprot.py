import evo_prot_grad
from transformers import AutoTokenizer, EsmForMaskedLM

def run_evo_prot_grad(raw_protein_sequence):
    #Convert raw protein seq to format expected by EvoProtGRAD
    fasta_format_sequence = f">Input_Sequence\n{raw_protein_sequence}"
    #Save FASTA string to a temp file 
    temp_fasta_path = "temp_input_sequence.fasta"
    with open(temp_fasta_path, "w") as file:
        file.write(fasta_format_sequence)
    #Load ESM2 model and tokenizer 
    esm2_expert = evo_prot_grad.get_expert(
    'esm',
    scoring_strategy = 'mutant_marginal',
    model=EsmForMaskedLM.from_pretrained("facebook/esm2_t30_150M_UR50D"),
    tokenizer=AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D"),
    temperature=0.95,
    device='cpu'
)
    #Initialise directed evolution 
    directed_evolution = evo_prot_grad.DirectedEvolution(
        wt_fasta=temp_fasta_path,
        output='best',  #best, last or all
        experts=[esm2_expert],
        parallel_chains=1,
        n_steps=20, #num of MCMC steps per chain 
        max_mutations=10,
        verbose=True
    )

    #Run evolution process
    variants, scores = directed_evolution()
    
    #Prints each variant and its associated score
    for variant, score in zip(variants, scores):
        print(f"Variant: {variant}, Score: {score}")

raw_protein_sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"  # Replace with your protein sequence
run_evo_prot_grad(raw_protein_sequence)

