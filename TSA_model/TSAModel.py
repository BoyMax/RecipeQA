import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids

# EmbeddingNet-ELMo
class ELMo(nn.Module):
    def __init__(self, word_hidden_size=100):
        super(ELMo, self).__init__()
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        if torch.cuda.is_available(): 
            self.elmo = Elmo(options_file, weight_file, 1, dropout=0.2, requires_grad = False).to(torch.device("cuda"))
        else:
            self.elmo = Elmo(options_file, weight_file, 1, dropout=0.2, requires_grad = False)
        self.lstm = nn.LSTM(1024, word_hidden_size, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, step):  #sentences: [['s1w1','s1w2','s1w3'],['s2w1','s2w2']] s:sentence w:word (batch_size, word_len)
        character_ids = batch_to_ids(step)
        if torch.cuda.is_available(): 
            character_ids = character_ids.cuda()
        embeddings = self.elmo(character_ids)['elmo_representations'][0] # embeddings:(batch, word_len, embed_dim) as (2, 3, 1024)
        output, (h_n, c_n) = self.lstm(embeddings)
        return torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) #(batch, 2*hidden_size)

class Hierarchy_Elmo_Net(nn.Module):
    def __init__(self, hidden_size=256, word_hidden_size=100):
        super(Hierarchy_Elmo_Net, self).__init__()
        self.embedding = ELMo(word_hidden_size)
        self.lstm = nn.LSTM(input_size=2*word_hidden_size, hidden_size=hidden_size, num_layers=1, bidirectional=True)
    def forward(self, texts): #texts (step_len, batch_size, word_len)
        batch = []
        for text in texts:
            embed_text = self.embedding(text) # shape of embed_text: (batch_size, word_hidden_size)
            if torch.cuda.is_available():
                batch.append(embed_text.cpu().detach().numpy())
            else:
                batch.append(embed_text.detach().numpy())
        # shape of batch: (step_len, batch_size, vector_dim)    
        if torch.cuda.is_available(): 
            batch = torch.Tensor(batch).cuda()
        else:
            batch = torch.Tensor(batch) # torch.Tensor(batch): covert list to tensor
        output,(h_n, c_n) = self.lstm(batch) 
        output = output.permute(1,0,2)
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        #output: (batch, seq_len, num_directions * hidden_size), h_n: (batch, hidden_size)
        return output, h_n

class Choice_ELMo_Net(nn.Module):
    def __init__(self, word_hidden_size):
        super(Choice_ELMo_Net, self).__init__()
        self.embedding = ELMo(word_hidden_size)
    def forward(self, choices): #choice (step_len, batch_size, word_len)
        embed_choices = []
        for choice in choices:
            embed_choice = self.embedding(choice) # shape of embed_text: (batch_size, word_len)
            if torch.cuda.is_available():
                embed_choices.append(embed_choice.cpu().detach().numpy())
            else:
                embed_choices.append(embed_choice.detach().numpy())
        # shape of batch: (step_len, batch_size, vector_dim)
        if torch.cuda.is_available(): 
            embed_choices = torch.Tensor(embed_choices).cuda()
        else:
            embed_choices = torch.Tensor(embed_choices) # torch.Tensor(batch): covert list to tensor
        return embed_choices.permute(1, 0, 2) # return shape(batch, step_len, vec_dim) consistent with the Doc2Vec

class Infersent(nn.Module):
    def __init__(self, c_features):
        super().__init__()
        self.linear1 = nn.Linear(4 * 2*c_features, 2*c_features)
        self.dropout = nn.Dropout(p = 0.2)
        self.linear2 = nn.Linear(2*c_features, 1)
    def forward(self, g, c):
        infersent_similarity = torch.tanh(self.linear1(torch.cat((g, c, torch.abs(g - c), g * c), 1)))
        infersent_similarity = self.dropout(infersent_similarity)
        infersent_similarity = self.linear2(infersent_similarity)
        return infersent_similarity

class HastyModel(nn.Module):
    # q_features: question features
    # c_features: choice features
    # c_features should equals to g_feature for comparing the similarity
    def __init__(self,q_features, g_features, c_features, 
                similarity_type, embed_hidden_size): 
        super(HastyModel, self).__init__()
        self.question = Hierarchy_Elmo_Net(q_features)
        self.choice = Choice_ELMo_Net(c_features)
        if similarity_type == 'cosine':
            self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        elif similarity_type == 'infersent':
            self.similarity = Infersent(c_features)
    # choices(batch, 4, word_len)
    def forward(self,questions, choices):
        # texts: (batch_size, step_len, word_len)  #questions: (batch_size, step_len, word_len)
        output, g = self.question(questions)
        # g (batch_size, c_dim) where g_dim = c_dim = embedding_dim
        choice_output = self.choice(choices)
        # choice_output: (batch_size, choice_len, c_dim)
        choice_len = choice_output.size()[1]
        similarity_scores = []
        for i in range(choice_len):
            choice_outputs= choice_output[:,i,:] #choice_output(batch_size, dim)
            similarity = self.similarity(g, choice_outputs) #similarity(batch_size)
            similarity_scores.append(similarity) # for accuracy 
        return similarity_scores
        # similarity_scores #(choice_len, batch)