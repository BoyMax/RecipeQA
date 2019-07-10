import torch
import torch.nn as nn
import torch.nn.functional as F
from Doc2Vec import load_model
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids

# EmbeddingNet-Doc2Vec
class Doc2Vec(nn.Module):
    def __init__(self):
        super(Doc2Vec, self).__init__()
        self.doc2vec = load_model("doc2vec_text_model")
    def forward(self, steps):  #sentences: [['s1w1','s1w2','s1w3'],['s2w1','s2w2']] s:sentence w:word (batch_size, sentence)
        embed_steps = []
        for step in steps:
            embed_step = self.doc2vec.infer_vector(step, alpha=0.025,steps=500)
            embed_steps.append(embed_step)
        embed_steps = np.array(embed_steps)  #(step_len, vector_dim) -- (2, 100)
        return embed_steps #(step_len, vector_dim)

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
        self.lstm = nn.LSTM(1024, word_hidden_size, num_layers=1, batch_first=True)

    def forward(self, step):  #sentences: [['s1w1','s1w2','s1w3'],['s2w1','s2w2']] s:sentence w:word (batch_size, word_len)
        character_ids = batch_to_ids(step)
        if torch.cuda.is_available(): 
            character_ids = character_ids.cuda()
        embeddings = self.elmo(character_ids)['elmo_representations'][0] # embeddings:(batch, word_len, embed_dim) as (2, 3, 1024)
        output, (h_n, c_n) = self.lstm(embeddings)
        return h_n[-1, :, :] #(batch, hidden_size)

class Hierarchy_Elmo_Net(nn.Module):
    def __init__(self, hidden_size=256, word_hidden_size=100):
        super(Hierarchy_Elmo_Net, self).__init__()
        self.embedding = ELMo(word_hidden_size)
        self.lstm = nn.LSTM(input_size=word_hidden_size, hidden_size=hidden_size, num_layers=1)
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
        #output: (batch, seq_len, num_directions * hidden_size), h_n: (num_layers * num_directions, batch, hidden_size)
        return output, h_n

class Hierarchy_Doc2Vec_Net(nn.Module):
    def __init__(self, hidden_size=512):
        super(Hierarchy_Doc2Vec_Net, self).__init__()
        self.embedding = Doc2Vec()
        self.lstm = nn.LSTM(input_size=100, hidden_size=hidden_size, num_layers=3, batch_first=True)
    def forward(self, texts): #texts (batch_size, step_len, word_len)
        batch = []
        for text in texts:
            embed_text = self.embedding(text) # shape of embed_text: (step_len, vector_dim)
            batch.append(embed_text)
        # shape of batch: (batch_size, step_len, vector_dim)    
        if torch.cuda.is_available(): 
            batch = torch.Tensor(batch).cuda()
        else:
            batch = torch.Tensor(batch) # torch.Tensor(batch): covert list to tensor
        output,(h_n, c_n) = self.lstm(batch) 
        #output: (batch, seq_len, num_directions * hidden_size), h_n: (num_layers * num_directions, batch, hidden_size)
        return output, h_n

class Choice_Doc2Vec_Net(nn.Module):
    def __init__(self):
        super(Choice_Doc2Vec_Net, self).__init__()
        self.embedding = Doc2Vec()
    def forward(self, choices): #texts (batch_size, step_len, word_len)
        batch = []
        for choice in choices:
            embed_choice = self.embedding(choice) # shape of embed_text: (step_len, vector_dim)
            batch.append(embed_choice)
        # shape of batch: (batch_size, step_len, vector_dim)
        if torch.cuda.is_available(): 
            batch = torch.Tensor(batch).cuda()
        else:
            batch = torch.Tensor(batch) # torch.Tensor(batch): covert list to tensor
        return batch

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

class HingeRankLoss(nn.Module):
    def __init__(self, margin, similarity_type='cosine', c_features=100):
        super().__init__()
        self.margin = margin
        if similarity_type == 'cosine':
            self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        elif similarity_type == 'infersent':
            self.similarity = Infersent(c_features)
    def forward(self, g, p, n):
        if torch.cuda.is_available(): 
            batch_zeros = torch.zeros(p.size()[0]).cuda()
        else:
            batch_zeros = torch.zeros(p.size()[0])
        loss = torch.max(batch_zeros, self.margin - self.similarity(g,p) + self.similarity(g,n))
        return torch.mean(loss)

class Infersent(nn.Module):
    def __init__(self, c_features):
        super().__init__()
        self.linear = nn.Linear(4 * c_features,1)
    def forward(self, g, c):
        infersent_similarity = torch.cat((g, c, torch.abs(g - c), g * c), 1)
        return self.linear(infersent_similarity)

class Attention(nn.Module):
    def __init__(self,d_features=512, q_features=512, m_features = 256, g_features=100, embedding_type='Doc2Vec', embed_hidden_size=100):
        super(Attention, self).__init__()
        if embedding_type == 'Doc2Vec':
            self.questionNet = Hierarchy_Doc2Vec_Net(q_features)
            self.textNet = Hierarchy_Doc2Vec_Net(d_features)
        elif embedding_type == 'ELMo':
            self.questionNet = Hierarchy_Elmo_Net(q_features, embed_hidden_size)
            self.textNet = Hierarchy_Elmo_Net(d_features, embed_hidden_size)
        self.r_dim = d_features # num_direction * d_features (unidirectional: num_direction=1)
        self.d_m = nn.Linear(in_features=d_features, out_features=m_features, bias=False)
        self.r_m = nn.Linear(in_features=self.r_dim, out_features=m_features, bias=False)
        self.q_m = nn.Linear(in_features=q_features, out_features=m_features, bias=False)
        self.m_s = nn.Linear(in_features=m_features, out_features=1, bias=False)
        self.r_r = nn.Linear(in_features=self.r_dim, out_features=self.r_dim, bias=False)
        self.r_g = nn.Linear(in_features=self.r_dim, out_features=g_features, bias=False)
        self.q_g = nn.Linear(in_features=q_features, out_features=g_features, bias=False)
        # input shape of nn.Linear(batch_size, in_features)
        # output shape of nn.Linear(batch_size, out_features)
    def forward(self, texts, questions):# text:(batch_size, step_len, word_len) question:(batch_size, step_len, word_len)
        text_output, text_h_n = self.textNet(texts) #output: (batch, seq_len, num_directions * hidden_size), h_n: (num_layers * num_directions, batch, hidden_size)
        question_output, question_h_n = self.questionNet(questions) #question_output: (batch, seq_len, num_directions * hidden_size)
        batch_size = question_output.size()[0]
        question_len = question_output.size()[1]
        if torch.cuda.is_available(): 
            r = torch.zeros(batch_size, 1, self.r_dim).cuda()
        else:
            r = torch.zeros(batch_size, 1, self.r_dim) # the 2nd dimension=1 because there is no step dimension in r.

        for i in range(question_len):
            m_i = torch.tanh(self.d_m(text_output) + self.r_m(r) + self.q_m(question_output[:,i,:]).unsqueeze(1)) #m_i(batch_size, step_len, m_dim)
            s_i = F.softmax(self.m_s(m_i), dim=1) # dim=1 means doing softman for steps 
            #s_i:(batch_size,step_len, s_dim)  text_output:(batch, seq_len, num_directions * hidden_size)
            # infered by r equation as follow: s_dim=1, r_dim = num_directions * hidden_size
            r = torch.matmul(s_i.permute(0,2,1), text_output) + torch.tanh(self.r_r(r)) # r(batch_size, 1, self.r_dim)
        g = torch.tanh(self.r_g(r).squeeze(1) + self.q_g(question_h_n[-1, :, :]))
        return g #(batch, g_dim) where g_dim = choice_dim

class SpatialAttention(nn.Module):
    def __init__(self,d_features=512, q_features=512, m_features = 256, g_features=100, embedding_type='ELMo', embed_hidden_size=256):
        super(SpatialAttention, self).__init__()
        if embedding_type == 'Doc2Vec':
            self.questionNet = Hierarchy_Doc2Vec_Net(q_features)
            self.textNet = Hierarchy_Doc2Vec_Net(d_features)
        elif embedding_type == 'ELMo':
            self.questionNet = Hierarchy_Elmo_Net(q_features, embed_hidden_size)
            self.textNet = Hierarchy_Elmo_Net(d_features, embed_hidden_size)
        self.r_dim = d_features # num_direction * d_features (unidirectional: num_direction=1)
        self.d_m = nn.Linear(in_features=d_features, out_features=m_features, bias=False)
        self.q_m = nn.Linear(in_features=q_features, out_features=m_features, bias=False)
        self.m_s = nn.Linear(in_features=m_features, out_features=1, bias=False)  # get spatial attention score between question and text
        self.r_g = nn.Linear(in_features=self.r_dim, out_features=g_features, bias=False)
        self.q_g = nn.Linear(in_features=q_features, out_features=g_features, bias=False)
        # input shape of nn.Linear(batch_size, in_features)
        # output shape of nn.Linear(batch_size, out_features)
    def forward(self, texts, questions):# text:(batch_size, step_len, word_len) question:(batch_size, step_len, word_len)
        text_output, text_h_n = self.textNet(texts) #output: (batch, seq_len, num_directions * hidden_size), h_n: (num_layers * num_directions, batch, hidden_size)
        question_output, question_h_n = self.questionNet(questions) #question_output: (batch, seq_len, num_directions * hidden_size)
        batch_size = question_output.size()[0]
        question_len = question_output.size()[1]
        m = torch.tanh(self.d_m(text_output) + self.q_m(question_h_n[-1, :, :].unsqueeze(1))) #m(batch_size, step_len, m_dim)
        s = F.softmax(self.m_s(m), dim=1) # dim=1 means doing softmax for steps, s is the attention score
        #s:(batch_size,step_len, 1)  text_output:(batch, seq_len, num_directions * hidden_size)
        r = torch.sum(s * text_output, 1) # s * text_output:(batch_size, step_len, num_directions * hidden_size), r means (weight_att * text)
        #r:(batch, num_directions * hidden_size)
        g = torch.tanh(self.r_g(r).squeeze(1) + self.q_g(question_h_n[-1, :, :])) #question_h_n: (num_layers * num_directions, batch, hidden_size)
        return g #(batch, g_dim) where g_dim = choice_dim #means the embedding between text and question

class TemporalAttention(nn.Module):
    def __init__(self,d_features=512, q_features=512, m_features = 256, g_features=100, embedding_type='ELMo', img_hidden_size=2048):
        super(TemporalAttention, self).__init__()
        self.imageNet = nn.LSTM(input_size=1000, hidden_size=img_hidden_size, num_layers=1)
        self.r_dim = d_features # num_direction * d_features (unidirectional: num_direction=1)
        self.d_m = nn.Linear(in_features=d_features, out_features=m_features, bias=False)
        self.q_m = nn.Linear(in_features=q_features, out_features=m_features, bias=False)
        self.m_s = nn.Linear(in_features=m_features, out_features=1, bias=False)  # get spatial attention score between question and text
        self.r_g = nn.Linear(in_features=self.r_dim, out_features=g_features, bias=False)
        self.q_g = nn.Linear(in_features=q_features, out_features=g_features, bias=False)
    def forward(self, images, questions):
        #images:(batch, img_len) images only contains the name of image
        return



class TSAModel(nn.Module):
    # d_features: document features
    # q_features: question features
    # c_features: choice features
    # m_features is in attention, g features is the output features of attention.
    # c_features should equals to g_feature for comparing the similarity
    def __init__(self,d_features=512, q_features=512, m_features = 256, g_features=100, c_features=100, 
                similarity_type = 'cosine', embedding_type='Doc2Vec', embed_hidden_size=100): 
        super(TSAModel, self).__init__()
        self.attention = SpatialAttention(d_features, q_features, m_features, g_features, embedding_type, embed_hidden_size)
        if embedding_type == 'Doc2Vec':
            self.choice = Choice_Doc2Vec_Net() # for embedding
            self.right_answer = Hierarchy_Doc2Vec_Net(q_features)
            self.wrong_answer = Hierarchy_Doc2Vec_Net(q_features)
        elif embedding_type == 'ELMo':
            self.choice = Choice_ELMo_Net(c_features)
            self.right_answer = Hierarchy_Elmo_Net(q_features, embed_hidden_size)
            self.wrong_answer = Hierarchy_Elmo_Net(q_features, embed_hidden_size)
        if similarity_type == 'cosine':
            self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        elif similarity_type == 'infersent':
            self.similarity = Infersent(c_features)
    # choices(batch, 4, word_len)
    def forward(self, texts, questions, choices):
        # texts: (batch_size, step_len, word_len)  #questions: (batch_size, step_len, word_len)
        g = self.attention(texts, questions) 
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