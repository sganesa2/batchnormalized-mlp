import torch
from typing import Optional 

from model.train import BatchNormalizedMLP

def run_batchnormlized_mlp(no_of_words:int, trained_model:BatchNormalizedMLP, initial_context:Optional[str]='...')->list[str]:
    """
    - Continues sampling the most probable character for a standard context of '...'
      from the trained model until it encounters a '.'
    - This is performed 'no_of_words' number of times.
    """
    #Set the model to eval mode
    trained_model.eval()

    stoi_dict, itos_dict = stoi(), itos()
    if len(initial_context)!=trained_model.n:
        raise ValueError(f"Context size !={trained_model.n}")
    
    words = []
    initial_context = [stoi_dict[c] for c in initial_context]
    
    for _ in range(no_of_words):
        word = "".join(map(lambda c: "" if c==0 else itos_dict[c], initial_context))
        context = initial_context
        idx = 1
        while itos_dict[idx]!='.':
            x = torch.tensor(context)
            logits = trained_model.forward(x)
            probs = logits.softmax(dim=0)
            idx = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [idx]
            word+=itos_dict[idx]
        words.append(word[:-1])

    return words

def stoi()->dict[str,int]:
    start_index, total_chars = 97, 26
    stoi_dict = {chr(i):i-start_index+1 for i in range(start_index, start_index+total_chars+1)}
    return {".":0, **stoi_dict}

def itos()->dict[int,str]:
    stoi_dict = stoi()
    return {v:k for k,v in stoi_dict.items()}

