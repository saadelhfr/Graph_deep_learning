import torch
import transformers
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEncoder
from tqdm import tqdm
import numpy as np
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomBidirectionalTransformer(nn.Module):
    def __init__(self, max_len , custom_sequence_parser, device = DEVICE ,embed_size=87, projection_size=768 ):
        super(CustomBidirectionalTransformer, self).__init__()
        self.max_len = max_len
        self.device = device
        self.embed_size = embed_size

        config = transformers.AutoConfig.from_pretrained('bert-base-uncased')
        self.embedding_projection = nn.Linear(embed_size, projection_size).to(self.device)


        # Define special token embeddings that are not learnable (mother, father, unknown, padding)

        self.special_embeddings = nn.Embedding(5, embed_size).to(self.device)
        self.mother_id, self.father_id, self.unknown_id, self.different_family_id , self.padding_id = 0, 1, 2, 3 , 4
        self.special_embeddings.weight.requires_grad = False



        # Define positional embeddings up to max_len
        self.position_embeddings = nn.Embedding(max_len, projection_size).to(self.device)
        self.position_embeddings.weight.requires_grad = False


        # Placeholder for the transformer layers
        self.transformer = BertEncoder(config).to(self.device)

        self.custom_sequence_parser = custom_sequence_parser

        self.LayerNorm = nn.LayerNorm(projection_size, eps=1e-12).to(self.device)
        self.dropout = nn.Dropout(0.1).to(self.device)

        self.output_projection = nn.Linear(projection_size, embed_size).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        self.scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau( self.optimizer , mode='min' , factor=0.5 , patience=5 , verbose=True)
    

    def add_special__embeddings(self, sequence):
        special_embeds = sequence
        pad_token_position = []
        mother_token , father_token , unknown_token , pad_token , new_family_token = self.custom_sequence_parser._create_special_tokens()
        mother_token = mother_token.to(self.device)
        father_token = father_token.to(self.device)
        unknown_token = unknown_token.to(self.device)
        pad_token = pad_token.to(self.device)
        new_family_token = new_family_token.to(self.device)
        

        for i , tensor in enumerate(sequence):

            tensor = tensor.to(self.device)
            if torch.eq(tensor , mother_token).all() :
                special_embeds[i] = self.special_embeddings(torch.tensor([self.mother_id]).to(self.device))
            elif torch.eq(tensor , father_token).all():
                special_embeds[i] = self.special_embeddings(torch.tensor([self.father_id]).to(self.device))
            elif torch.eq(tensor , unknown_token).all():
                special_embeds[i] = self.special_embeddings(torch.tensor([self.unknown_id]).to(self.device))
            elif torch.eq(tensor , new_family_token).all() :
                special_embeds[i] = self.special_embeddings(torch.tensor([self.different_family_id]).to(self.device))
            elif torch.eq(tensor , pad_token).all():
                pad_token_position.append(i)
                special_embeds[i] = self.special_embeddings(torch.tensor([self.padding_id]).to(self.device))
            else :
                special_embeds[i] = tensor.unsqueeze(0)
            
        
        return special_embeds , pad_token_position 



    def pad_mask(self, mask, pad_token_position):
        # make zeors on pad token position
        for i in pad_token_position:
            mask[i] = 0
        return mask
        
    def mask_tensors(self, input_embeddings , pad_token_position , mask_prob=0.40):

            zeros = torch.zeros(input_embeddings[0].shape).to(self.device)
            
            total_positions = len(input_embeddings) - len(pad_token_position)
            num_masked = int(mask_prob * total_positions)

            # Randomly select positions to mask
            masked_positions = torch.randperm(total_positions)[:num_masked].to(self.device)
            
            # masked_positions = random.sample(real_embed_position, num_masked)
            # masked_positions = torch.tensor(masked_positions).to(self.device)
            original_embeddings = torch.zeros(num_masked , self.embed_size).to(self.device)
            # Store the original embeddings for these positions
            for i   in range(num_masked):
                original_embeddings[i] = input_embeddings[masked_positions[i]]
            
            # Replace the embeddings at these positions with zeros (or any other strategy you prefer)
            for pos in masked_positions:
                input_embeddings[pos] = zeros
            return input_embeddings, original_embeddings, masked_positions

        

    def _create_attention_mask(self, final_embeddings , pad_token_position, target_position=None):
        # Create a mask of 1s for tokens that are NOT MASKED, and 0s for tokens that are
        mask = torch.ones(final_embeddings.size(0), dtype=torch.long).to(self.device)
        if target_position is not None:
            mask[target_position] = 0
        for i in pad_token_position:
            mask[i] = 0
        return mask

    def _add_positional_embeddings(self, embeddings , pad_token_position):
        nbr_padds = len(pad_token_position)
        position_ids = torch.arange(0, embeddings.size(0) - nbr_padds).unsqueeze(0).to(self.device)
        position_embeds = self.position_embeddings(position_ids).squeeze(0)

        
        #complete it with zeros to be max_len
        position_embeds = torch.cat((position_embeds, torch.zeros(nbr_padds, position_embeds.size(1)).to(self.device)), dim=0)
        return embeddings + position_embeds

    def prepare_inputs_for_training(self, input_embeddings, target_position : int =None, mask_prob : float = 0.50 , batch_size : int = 100):
        input_tensor = []
        original_tensor = []
        masked_positions_for_train = []
        attention_masks = []
        pad_token_positions = []
        for sequence in tqdm(input_embeddings , total = len(input_embeddings) , desc = "preparing inputs for training") : 
            special_sequence , pad_token_position = self.add_special__embeddings(sequence)
            # create the attention mask
            special_sequence = torch.stack(special_sequence).to(self.device)
            attention_mask = self._create_attention_mask(special_sequence,pad_token_position, target_position)
            padded_attention_mask = self.pad_mask(attention_mask, pad_token_position)
            # add positional embeddings
            special_sequence, original_embeddings, masked_position = self.mask_tensors(special_sequence, pad_token_position,mask_prob)
            input_tensor.append(special_sequence)
            original_tensor.append(original_embeddings)
            masked_positions_for_train.append(masked_position)
            attention_masks.append(padded_attention_mask)
            pad_token_positions.append(pad_token_position)

        # make batches 
        batched_inputs = [torch.stack(input_tensor[i:i + batch_size]).squeeze(2).to(self.device) for i in range(0, len(input_tensor), batch_size)]
        batched_attention_masks = [torch.stack(attention_masks[i : i+batch_size]).to(self.device) for i in range(0, len(attention_masks), batch_size)]
        batched_original_embeddings = [original_tensor[i : i+batch_size] for i in range(0, len(original_tensor), batch_size)]
        batched_masked_positions = [masked_positions_for_train[i : i+batch_size] for i in range(0, len(masked_positions_for_train), batch_size)]
        batched_pad_token_positions = [pad_token_positions[i : i+batch_size] for i in range(0, len(pad_token_positions), batch_size)]

        # split into train and test
        train_input_tensor = batched_inputs[:int(len(batched_inputs)*0.8)]
        train_attention_masks = batched_attention_masks[:int(len(batched_attention_masks)*0.8)]
        train_original_embeddings = batched_original_embeddings[:int(len(batched_original_embeddings)*0.8)]
        train_masked_positions = batched_masked_positions[:int(len(batched_masked_positions)*0.8)]
        train_pad_token_positions = batched_pad_token_positions[:int(len(batched_pad_token_positions)*0.8)]

        test_input_tensor = batched_inputs[len(train_input_tensor):]
        test_attention_masks = batched_attention_masks[len(train_attention_masks):]
        test_original_embeddings = batched_original_embeddings[len(train_original_embeddings):]
        test_masked_positions = batched_masked_positions[len(train_masked_positions):]
        test_pad_token_positions = batched_pad_token_positions[len(train_pad_token_positions):]

        return train_input_tensor, train_attention_masks, train_original_embeddings, train_masked_positions, train_pad_token_positions, test_input_tensor, test_attention_masks, test_original_embeddings, test_masked_positions, test_pad_token_positions

    def train_forward_one_batch (self,input_tensor , attention_masks , original_embeddings , masked_positions , pad_token_positions):
        batch_size = input_tensor.shape[0]
        input_tensor = input_tensor.view(-1, input_tensor.size(-1))
        # paass the input_tensor through projection , layer norm and dropout
        input_tensor = self.embedding_projection(input_tensor)
        input_tensor = self.LayerNorm(input_tensor)
        input_tensor = self.dropout(input_tensor)

        # reshape the input tensor
        input_tensor = input_tensor.view(batch_size, -1, input_tensor.size(-1))
        # add positional embeddings
        for i in range(input_tensor.shape[0]):
            input_tensor[i] = self._add_positional_embeddings(input_tensor[i].squeeze(0), pad_token_positions[i])

        
        # pass the input tensor through the transformer
        outputs = self.transformer(input_tensor, attention_mask = attention_masks.unsqueeze(1).unsqueeze(2))
        # pass the output through the linear layer
        output = outputs.last_hidden_state
        # flatten the output
        output = output.view(-1, output.size(-1))
        output = self.output_projection(output)
        # reshape the output
        output = output.view(batch_size, -1, output.size(-1))
        # delete all the unecesaary tensors
        torch.cuda.empty_cache()
       

        return output, original_embeddings, masked_positions
        





    def training_loop(self , inputs , epochs : int = 10, batch_size : int = 100, mask_prob : float = 0.50 , ):
        
        # get the train data 
        cosine_loss = nn.CosineEmbeddingLoss()
        mseLoss = nn.MSELoss()

        train_input_tensor, train_attention_masks, train_original_embeddings, train_masked_positions, train_pad_token_positions,  test_input_tensor, test_attention_masks, test_original_embeddings, test_masked_positions, test_pad_token_positions = self.prepare_inputs_for_training(inputs, mask_prob=mask_prob, batch_size=batch_size)


        # train the model
        lowest_loss = np.inf


        for epoch in range(epochs):
            epoch_loss = 0
            for i in tqdm(range(len(train_input_tensor)) , total =len(train_input_tensor) , desc= "epoch"+str(epoch)):
                self.optimizer.zero_grad()
                output, original_embeddings, masked_positions = self.train_forward_one_batch(train_input_tensor[i], train_attention_masks[i], train_original_embeddings[i], train_masked_positions[i], train_pad_token_positions[i])
                batch_loss = 0
                for i in range(len(original_embeddings)):
                    originals = original_embeddings[i].squeeze(1)
                    masked_position = masked_positions[i]
                    outputs = output[i][masked_position]
                    loss_mse = mseLoss(outputs, originals)
                    loss_cosine = cosine_loss(outputs, originals, torch.ones(outputs.shape[0]).to(self.device))
                    batch_loss +=0.6*loss_cosine + 0.4*loss_mse
                batch_loss = batch_loss/len(original_embeddings)
                
                batch_loss.backward()
                epoch_loss += batch_loss
                self.optimizer.step()
                torch.cuda.empty_cache()
            epoch_loss = epoch_loss/len(train_input_tensor)
            self.scheduler.step(epoch_loss)
            torch.cuda.empty_cache()



            print(f"epoch loss {epoch}: ", epoch_loss.item())
            if epoch_loss.item() < lowest_loss:
                lowest_loss = epoch_loss.item()
                torch.save(self.state_dict(), "bestmodel.pth")
                print("best model saved")
                # save the loss of the model 
                with open("../loss.txt", "w") as f:
                    f.write(str(lowest_loss))
                
                patience = 0
            else : 
                patience +=1
            if patience == 10:
                print("early stopping")
                break
        print("training complete")
        print("testing the model")
        # test the model 
        inputs_for_testing = ( test_input_tensor, test_attention_masks, test_original_embeddings, test_masked_positions, test_pad_token_positions)
        mean_batch_loss, mean_other_loss= self.test_loop(inputs_for_testing, batch_size=batch_size, mask_prob=mask_prob , no_prep = True)

        # ask the user if he wants to save the model
        save = input("Do you want to save the model? (y/n)")
        if save == "y":
            torch.save(self.state_dict(), "../model.pth")
            print("model saved")
        else:
            print("model not saved")

        return mean_batch_loss, mean_other_loss
            



    def test_loop(self , inputs , batch_size : int = 100, mask_prob : float = 0.50 ,no_prep :bool = True):
        if no_prep:
            test_input_tensor, test_attention_masks, test_original_embeddings, test_masked_positions, test_pad_token_positions = inputs
        else:
            _, _, _, _, _, test_input_tensor, test_attention_masks, test_original_embeddings, test_masked_positions, test_pad_token_positions = self.prepare_inputs_for_training(inputs, mask_prob=mask_prob, batch_size=batch_size)
        cosine_loss = nn.CosineEmbeddingLoss()
        mseLoss = nn.MSELoss()
        
        total_batch_losses = []  # Store batch losses for each batch
        total_other_losses = []  # Store the other embeddings losses for each batch
        
        for i in tqdm(range(len(test_input_tensor)), total=len(test_input_tensor), desc="testing"):
            output, original_embeddings, masked_positions = self.train_forward_one_batch(test_input_tensor[i], test_attention_masks[i], test_original_embeddings[i], test_masked_positions[i], test_pad_token_positions[i])
            
            batch_loss = []  # Initialize batch_loss
            other_losses = []  # List to store losses of other embeddings
            
            for j in range(len(original_embeddings)):
                originals = original_embeddings[j].squeeze(1)
                masked_position = masked_positions[j]
                outputs = output[j][masked_position]
                other_outputs = output[j][~masked_position]
                loss_mse = mseLoss(outputs, originals)
                target = torch.ones(outputs.shape[0]).to(self.device)
                try :
                    loss_cosine = cosine_loss(outputs, originals, target)
                except:
                    loss_cosine = cosine_loss(outputs, originals, 1)
                batch_loss.append( 0.6*loss_cosine + 0.4*loss_mse)

                # Compute losses for other embeddings
                for other, original in zip(other_outputs, originals):
                    loss_mse_other = mseLoss(other, original)
                    other = other.unsqueeze(0)
                    original = original.unsqueeze(0)
                    target = torch.ones(other.shape[0]).to(self.device)
                    try :
                        loss_cosine_other = cosine_loss(outputs, originals, target) 
                    except:
                        loss_cosine_other = cosine_loss(outputs, originals, torch.tensor([1]).to(self.device))
                    other_losses.append(0.6*loss_cosine_other + 0.4*loss_mse_other)
            
            total_batch_losses.append(torch.mean(torch.tensor(batch_loss)))  # Get mean loss of this batch and append
            total_other_losses.append(torch.mean(torch.tensor(other_losses)))  # Get mean other loss of this batch and append

        mean_batch_loss = total_batch_losses
        mean_other_loss = total_other_losses
        
        return mean_batch_loss, mean_other_loss

    def pad_sequence(self , sequence : torch.tensor):
        # pad the sequence to the max length
        attention_mask = torch.ones(self.max_len).to(self.device)
        print(sequence.shape[0])
        for i in range(sequence.shape[0],  self.max_len):
            attention_mask[i] = 0
            
        if sequence.shape[0] < self.max_len:
            pad = self.special_embeddings(torch.tensor([self.padding_id]).to(self.device))
            padding = pad.repeat(self.max_len - sequence.shape[0], 1)
            sequence = torch.cat((sequence, padding), dim=0)
        return sequence , attention_mask

    def forward_one_sequence(self , sequence):
        # pass the sequence through the model
        sequence , attention_mask= self.pad_sequence(sequence)
        sequence = sequence.unsqueeze(0) 
        sequence = self.embedding_projection(sequence)
        sequence = self.LayerNorm(sequence)
        sequence = self.dropout(sequence)
        sequence_outputs = self.transformer(sequence, attention_mask=attention_mask)
        output = sequence_outputs.last_hidden_state
        output = self.output_projection(output)

        return output 


    def predict(self , embedding : torch.tensor):
        


        mother_tensor = self.special_embeddings(torch.tensor([self.mother_id]).to(self.device)).to(self.device)
        father_tensor = self.special_embeddings(torch.tensor([self.father_id]).to(self.device)).to(self.device)
        embedding = embedding.to(self.device)

        zeros = torch.zeros_like(embedding).to(self.device)
      
        predict_father_sequence = torch.cat(( zeros , father_tensor, embedding  ), dim=0)
        
        predict_mother_sequence = torch.cat(( zeros , mother_tensor, embedding ), dim=0)
        
        # pad the sequences to the max length
        # pass the sequences through the model
        predicted_father_sequence = self.forward_one_sequence(predict_father_sequence)
        predicted_mother_sequence = self.forward_one_sequence(predict_mother_sequence)

        predicted_father = predicted_father_sequence[0]
        predicted_mother = predicted_mother_sequence[0]

        return predicted_father, predicted_mother

 

      