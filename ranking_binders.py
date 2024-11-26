from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

#Load base model & tokeniser 
base_model_path = "facebook/esm2_t12_35M_UR50D"
model = AutoModelForMaskedLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

model.eval()

#Define protein of interest & potential binders 
protein_of_interest = "MLTEVMEVWHGLVIAVVSLFLQACFLTAINYLLSRHMAHKSEQILKAASLQVPRPSPGHHHPPAVKEMKETQTERDIPMSDSLYRHDSDTPSDSLDSSCSSPPACQATEDVDYTQVVFSDPGELKNDSPLDYENIKEITDYVNVNPERHKPSFWYFVNPALSEPAEYDQVAM"
potential_binders = [
    "MASPGSGFWSFGSEDGSGDSENPGTARAWCQVAQKFTGGIGNKLCALLYGDAEKPAESGGSQPPRAAARKAACACDQKPCSCSKVDVNYAFLHATDLLPACDGERPTLAFLQDVMNILLQYVVKSFDRSTKVIDFHYPNELLQEYNWELADQPQNLEEILMHCQTTLKYAIKTGHPRYFNQLSTGLDMVGLAADWLTSTANTNMFTYEIAPVFVLLEYVTLKKMREIIGWPGGSGDGIFSPGGAISNMYAMMIARFKMFPEVKEKGMAALPRLIAFTSEHSHFSLKKGAAALGIGTDSVILIKCDERGKMIPSDLERRILEAKQKGFVPFLVSATAGTTVYGAFDPLLAVADICKKYKIWMHVDAAWGGGLLMSRKHKWKLSGVERANSVTWNPHKMMGVPLQCSALLVREEGLMQNCNQMHASYLFQQDKHYDLSYDTGDKALQCGRHVDVFKLWLMWRAKGTTGFEAHVDKCLELAEYLYNIIKNREGYEMVFDGKPQHTNVCFWYIPPSLRTLEDNEERMSRLSKVAPVIKARMMEYGTTMVSYQPLGDKVNFFRMVISNPAATHQDIDFLIEEIERLGQDL", 
    "MAAGVAGWGVEAEEFEDAPDVEPLEPTLSNIIEQRSLKWIFVGGKGGVGKTTCSCSLAVQLSKGRESVLIISTDPAHNISDAFDQKFSKVPTKVKGYDNLFAMEIDPSLGVAELPDEFFEEDNMLSMGKKMMQEAMSAFPGIDEAMSYAEVMRLVKGMNFSVVVFDTAPTGHTLRLLNFPTIVERGLGRLMQIKNQISPFISQMCNMLGLGDMNADQLASKLEETLPVIRSVSEQFKDPEQTTFICVCIAEFLSLYETERLIQELAKCKIDTHNIIVNQLVFPDPEKPCKMCEARHKIQAKYLDQMEDLYEDFHIVKLPLLPHEVRGADKVNTFSALLLEPYKPPSAQ", 
    "EKTGLSIRGAQEEDPPDPQLMRLDNMLLAEGVSGPEKGGGSAAAAAAAAASGGSSDNSIEHSDYRAKLTQIRQIYHTELEKYEQACNEFTTHVMNLLREQSRTRPISPKEIERMVGIIHRKFSSIQMQLKQSTCEAVMILRSRFLDARRKRRNFSKQATEILNEYFYSHLSNPYPSEEAKEELAKKCSITVSQSLVKDPKERGSKGSDIQPTSVVSNWFGNKRIRYKKNIGKFQEEANLYAAKTAVTAAHAVAAAVQNNQTNSPTTPNSGSSGSFNLPNSGDMFMNMQSLNGDSYQGSQVGANVQSQVDTLRHVINQTGGYSDGLGGNSLYSPHNLNANGGWQDATTPSSVTSPTEGPGSVHSDTSN"
]

def compute_mlm_loss(protein, binder, interations = 3): 
    total_loss = 0.0
    for _ in range(iterations):
        concatenated_sequence = protein + ":" + binder
        tokens = list(concatenated_sequence)
        mask_rate = 0.15
        num_mask = int(len(tokens) * mask_rate)
        available_indices = [i for i, token in enumerate(tokens) if token != ":"]
        probs = torch.ones(len(available_indices))
        mask_indices = torch.multinomial(probs, num_mask, replacement=False)

        for idx in mask_indices:
            tokens[available_indices[idx]] = tokenizer.mask_token
        masked_sequence = "".join(tokens)
        inputs = tokenizer(masked_sequence, return_tensors="pt", truncation=True, max_length=1024, padding='max_length')

        # Compute the MLM loss
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        total_loss += loss.item()
    return total_loss / iterations

#Compute MLM losses for each potential binder 
mlm_losses = {}
for binder in potential_binders:
    loss = compute_mlm_loss(protein_of_interest, binder)
    mlm_losses[binder] = loss

#Rank binders based on MLM loss
ranked_binders = sorted(mlm_losses, key=mlm_losses.get)

print("Ranking of Potential Binders:")
for idx, binder in enumerate(ranked_binders, 1):
    print(f"{idx}. {binder} - MLM Loss: {mlm_losses[binder]}")

