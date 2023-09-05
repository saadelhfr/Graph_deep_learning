import torch


class CustomSequenceParser():
    def __init__(self ,max_len , sequences , embed_size , device ):
        self.sequences = sequences
        self.max_len = max_len
        self.embed_size = embed_size
        self.device = device


    def _create_special_tokens(self):
        mother_token = torch.zeros(self.embed_size).to(self.device)
        father_token = torch.ones(self.embed_size).to(self.device)
        unknown_token = torch.ones(self.embed_size) * 2
        unknown_token.to(self.device)
        pad_token = torch.ones(self.embed_size) * 3
        pad_token.to(self.device)
        new_family_token = torch.ones(self.embed_size) * 4
        new_family_token.to(self.device)
        return mother_token , father_token , unknown_token, pad_token , new_family_token

    def add_special_tokens(self , sequence):
        mother_token , father_token , unknown_token , pad_token , new_family_token = self._create_special_tokens()
        sequence_with_special_tokens = [new_family_token]
        for i , tensor in enumerate(sequence):
            sequence_with_special_tokens.append(tensor)
            if i != len(sequence) - 1 :
                if tensor[0] == 1 :
                    sequence_with_special_tokens.append(mother_token)
                elif tensor[0] == -1 :
                    sequence_with_special_tokens.append(father_token)
                else :
                    sequence_with_special_tokens.append(unknown_token)
            

            
        sequence_with_special_tokens.append(new_family_token)
        if len(sequence_with_special_tokens) < self.max_len :
            sequence_with_special_tokens.extend([pad_token] * (self.max_len - len(sequence_with_special_tokens)))

        return sequence_with_special_tokens[:self.max_len]