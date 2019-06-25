import torch
import torch.nn as nn
import torch.nn.functional as F
from Doc2Vec import load_model
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# QuestionNet
class QuestionNet(nn.Module):
    def __init__(self, hidden_size):
        super(QuestionNet, self).__init__()
        self.embedding = Doc2Vec()
        self.lstm = nn.LSTM(input_size=100, hidden_size=hidden_size, num_layers=3, batch_first=True)
    def forward(self, questions): #questions (batch_size, step_len, word_len)
        batch = []
        for question in questions:
            embed_question = self.embedding(question) # shape of embed_question: (step_len, vector_dim)
            batch.append(embed_question)
        # shape of batch(batch_size, step_len, vector_dim)
        if torch.cuda.is_available(): 
            batch = torch.Tensor(batch).cuda()
        else:
            batch = torch.Tensor(batch) # torch.Tensor(batch): covert list to tensor
        output,(h_n, c_n) = self.lstm(batch) 
        #output: (batch, step_len, num_directions * hidden_size), h_n: (num_layers * num_directions, batch, hidden_size)
        return output, h_n

class TextNet(nn.Module):
    def __init__(self, hidden_size=512):
        super(TextNet, self).__init__()
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

class Attention(nn.Module):
    def __init__(self,d_features=512, q_features=512, m_features = 256, g_features=100):
        super(Attention, self).__init__()
        self.questionNet = QuestionNet(q_features)
        self.textNet = TextNet(d_features)
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

class ChoiceNet(nn.Module):
    def __init__(self, hidden_size):
        super(ChoiceNet, self).__init__()
        self.embedding = Doc2Vec()
        #self.lstm = nn.LSTM(input_size=100, hidden_size=hidden_size, num_layers=3, batch_first=True)
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

class ImpatientReaderModel(nn.Module):
    # d_features: document features
    # q_features: question features
    # c_features: choice features
    # m_features is in attention, g features is the output features of attention.
    # c_features should equals to g_feature for comparing the similarity
    def __init__(self,d_features=512, q_features=512, m_features = 256, g_features=100, c_features=100, similarity_type = 'cosine'): 
        super(ImpatientReaderModel, self).__init__()
        self.attention = Attention(d_features, q_features, m_features, g_features)
        self.choice = ChoiceNet(c_features) # for embedding
        self.right_answer = QuestionNet(q_features)
        self.wrong_answer = QuestionNet(q_features)
        if similarity_type == 'cosine':
            self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        elif similarity_type == 'infersent':
            self.similarity = Infersent(c_features)
    # choices(batch, 4, word_len)
    # replaced_questions(2, batch, 4,word_len), replaced_questions[0,:,:,:] means right answers, replaced_questions[1,:,:,:] means wrong answers.
    def forward(self, texts, questions, choices, replaced_questions):
        # texts: (batch_size, step_len, word_len)  #questions: (batch_size, step_len, word_len)
        g = self.attention(texts, questions) 
        # output(batch_size, c_dim) where g_dim = c_dim = embedding_dim
        choice_output = self.choice(choices)
        # r_h_n (num_layers * num_directions, batch, hidden_size)
        r_o, r_h_n = self.right_answer(replaced_questions[0])
        w_o, w_h_n = self.wrong_answer(replaced_questions[1])
        # choice_output: (batch_size, choice_len, c_dim)
        choice_len = choice_output.size()[1]
        similarity_scores = []
        for i in range(choice_len):
            choice_outputs= choice_output[:,i,:] #choice_output(batch_size, dim)
            similarity = self.similarity(g, choice_outputs) #similarity(batch_size)
            similarity_scores.append(similarity) # for accuracy 
        return similarity_scores, g, r_h_n[-1,:,:], w_h_n[-1,:,:]
        # similarity_scores #(choice_len, batch)

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
        return self.linear(torch.cat((g, c, torch.abs(g - c), g * c), 1))