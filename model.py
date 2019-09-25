
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM

import numpy as np
from random import randint

def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight) 
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    if seq.size(1) == 1:
        return s
    return s.squeeze()



def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    #if len(seq.size()) == 2:
    #    seq = seq.unsqueeze(1)
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()



def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0)


class DocAttNet(nn.Module):
    def __init__(self, sent_hidden_size=784, doc_hidden_size=256,  num_classes=2):
        super(DocAttNet, self).__init__()

        self.weight_W_word = nn.Parameter(torch.Tensor(2* doc_hidden_size,2*doc_hidden_size))
        self.bias_word = nn.Parameter(torch.Tensor(2* doc_hidden_size,1))
        self.weight_proj_word = nn.Parameter(torch.Tensor(2*doc_hidden_size, 1))

        self.gru = nn.LSTM(sent_hidden_size, doc_hidden_size, bidirectional=True)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.bias_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1,0.1)
        self.softmax_word = nn.Softmax(dim=0)



    def forward(self, x):
        #input should be #[seq, batch]
        #self.gru.flatten_parameters()
        output_word, state_word = self.gru(x)

        word_squish = batch_matmul_bias(output_word, self.weight_W_word,self.bias_word, nonlinearity='tanh')
        word_attn = batch_matmul(word_squish, self.weight_proj_word)
        word_attn_norm = self.softmax_word(word_attn)
        word_attn_vectors = attention_mul(output_word, word_attn_norm)   

        return word_attn_vectors, word_attn_norm, output_word #state_word, word_attn_norm 

class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, num_labels=2):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.hidden_size = config.hidden_size
        self.mem_size = 512
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.att = DocAttNet(sent_hidden_size=config.hidden_size, doc_hidden_size = self.mem_size, num_classes = num_labels)


        self.classifier = torch.nn.Linear(self.mem_size *2, num_labels)
        self.classifier2 = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward2(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_id, token_type_id, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier2(pooled_output)
        return logits


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, long_doc=True):

        #import pdb; pdb.set_trace()
        if long_doc:
            #self.freeze_bert_encoder()
            zs = []
            for i in range(input_ids.shape[1]):
                _, pooled_output = self.bert(input_ids[:,i], token_type_ids[:,i], attention_mask[:,i], output_all_encoded_layers=False)
                #pooled_output = self.dropout(pooled_output)
                zs.append(pooled_output.detach())

            mem = torch.zeros(2, input_ids.shape[0], self.mem_size).cuda()

            attention_output, word_attn_norm = self.att( torch.stack(zs, 0), mem)
            attention_output = self.dropout(attention_output)
            logits = self.classifier(attention_output)
            return logits, word_attn_norm
        else:
            #self.unfreeze_bert_encoder()
            _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier2(pooled_output)
            return logits

        
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True



class MyBertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, num_labels=2):
        super(MyBertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier2 = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
        self.leaky = nn.LeakyReLU(0.2)



    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        x      = self.leaky(self.classifier1(pooled_output))
        logits = self.classifier2(x)

        
        return logits


class OldLongBert(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, num_labels=2):
        super(LongBert, self).__init__(config)
        self.num_labels = num_labels
        self.mem_size = config.hidden_size
        
        self.bert = BertModel(config)
        #self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.gru = nn.GRUCell( config.hidden_size,  config.hidden_size)
        self.classifier1 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier2 = torch.nn.Linear(config.hidden_size, num_labels)


        self.apply(self.init_bert_weights)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, hx, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        #pooled_output = self.dropout(pooled_output)
        hx     = self.gru(pooled_output, hx)
        x      = self.leaky(self.classifier1(hx))
        logits = self.classifier2(x)

        
        return logits, hx


class NormalBert(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, num_labels=2):
        super(NormalBert, self).__init__(config)
        self.num_labels = num_labels
        
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = torch.nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids, input_mask):
        _, x    = self.bert(input_ids[:,:256], segment_ids[:,:256], input_mask[:,:256], output_all_encoded_layers=False)
        x       = self.dropout(x)
        logits  = self.classifier1(x)

        
        return logits

class AttentionSource(nn.Module):
   
    def __init__(self, dim):
        super(AttentionSource, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)

    def forward(self, output, context):
        batch_size = output.size(1)
        hidden_size = output.size(2)
        input_size = context.size(0)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn

class NewAttention(nn.Module):
    def __init__(self, sent_hidden_size=784):
        super(NewAttention, self).__init__()

        self.rnn = nn.LSTM(sent_hidden_size, sent_hidden_size, batch_first=True , bidirectional=True)
        self.attention  = AttentionSource(sent_hidden_size)

        self.word_attention = nn.Linear(2 * sent_hidden_size, sent_hidden_size)

        # Word context vector to take dot-product with
        self.word_context_vector = nn.Linear(sent_hidden_size, 1, bias=False)


    def forward(self, x):
        #self.gru.flatten_parameters()
        rnn_output, hidden = self.rnn(x.transpose(0,1))
        # Apply the word-level RNN over the word embeddings (PyTorch automatically applies it on the PackedSequence)

        # Find attention vectors by applying the attention linear layer on the output of the RNN
        att_w = self.word_attention(rnn_output)  # (n_words, att_size)
        att_w = torch.tanh(att_w)  # (n_words, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(2)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  # (n_words)

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # (n_sentences, max(words_per_sentence))

        # Find sentence embeddings
        sentences = rnn_output * word_alphas.unsqueeze(2)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        sentences = sentences.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)

        return sentences, word_alphas, rnn_output.transpose(0,1)


class AttentionLongBert(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, sequence_len, input_len, num_labels=2):
        super(AttentionLongBert, self).__init__(config)
        self.num_labels = num_labels
        self.mem_size = config.hidden_size
        self.sequence_len = sequence_len
        self.total_input_len = sequence_len * input_len
        
        self.bert = BertModel(config)
        self.bert_layers = 12
        
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.dropout25 = torch.nn.Dropout(0.25)
        #self.rnn = nn.GRU( config.hidden_size,  config.hidden_size * 2, bidirectional = True)
        
        self.lstm = nn.LSTM( config.hidden_size,  config.hidden_size * 2, bidirectional = True)
        self.classifier1 = torch.nn.Linear(config.hidden_size * 9, num_labels)
        self.classifier8 = torch.nn.Linear(config.hidden_size * 8, num_labels)
        self.attention1 = torch.nn.Linear(self.sequence_len, 64)
        self.attention2 = torch.nn.Linear(64, 128)
        self.attention3 = torch.nn.Linear(128 + config.hidden_size, 2*config.hidden_size)
        #self.classifier10 = torch.nn.Linear(config.hidden_size * 10, num_labels)
        #self.classifier12_1 = torch.nn.Linear(config.hidden_size * 12, config.hidden_size * 4)
        #self.classifier12_2 = torch.nn.Linear(config.hidden_size * 4, num_labels)
        #self.bn = nn.BatchNorm1d(config.hidden_size * 9)

        self.apply(self.init_bert_weights)
        self.leaky = nn.LeakyReLU(0.2)

        #self.att = DocAttNet(sent_hidden_size=config.hidden_size, doc_hidden_size = self.mem_size, num_classes = num_labels)
        self.att = NewAttention(config.hidden_size)


    def forward(self, input_ids, segment_ids, input_mask):
        assert segment_ids.shape[1] == self.total_input_len
        zs = []
        #if self.training:
        #    self.att.rnn.flatten_parameters()
        self.att.rnn.flatten_parameters()
        
        input_ids   = input_ids.view(input_ids.size(0), self.sequence_len, -1)
        segment_ids = segment_ids.view(segment_ids.size(0), self.sequence_len, -1)
        input_mask  = input_mask.view(input_mask.size(0), self.sequence_len, -1)

        for i in range(self.sequence_len):
            z = self.bert(input_ids[:,i], segment_ids[:,i], input_mask[:,i], output_all_encoded_layers=False)[1].detach()
            zs.append(z)

        zs = torch.stack(zs)

        attention_output, norms, full_rnn_h = self.att(zs)
        h = full_rnn_h.view(self.sequence_len, -1, 2, self.mem_size)
        #TODO: try argmax without first_z
        norm_index = torch.argmax(norms, dim = 1)

        att_id = [] # torch.zeros([input_ids.size(0),256]).cuda()
        att_seg = [] #torch.zeros([input_ids.size(0),256]).cuda()
        att_mask = [] #torch.zeros([input_ids.size(0),256]).cuda()

        for batch_index in range(input_ids.size(0)):
            i = norm_index[batch_index]
            att_id.append( input_ids[batch_index,i])
            att_seg.append(segment_ids[batch_index,i])
            att_mask.append(input_mask[batch_index,i])

        attention_z = self.bert(torch.stack(att_id), torch.stack(att_seg), torch.stack(att_mask), output_all_encoded_layers=False)[1]

        
        #pass norm index into fc_layers
        a = self.leaky(self.attention1(norms))
        a = self.leaky(self.attention2(a))
        #a = self.attention3(torch.cat([a,attention_z], dim = 1))
        a = self.attention3(torch.cat([a,attention_z], dim = 1))

        first_z = self.bert(input_ids[:,0], segment_ids[:,0], input_mask[:,0], output_all_encoded_layers=False)[1]

        #x       = torch.cat([first_z, x[0,:,1], x[1,:,0], x[-1,:,0],attention_output,a], dim = 1)
        #x       = torch.cat([first_z, h[0,:,1], h[1,:,0], h[-1,:,0], attention_output,a], dim = 1)
        x       = torch.cat([first_z, full_rnn_h[1],full_rnn_h[-1], attention_output,a], dim = 1)
        x       = self.dropout(x)

        #logits  = self.classifier8(x)
        logits  = self.classifier1(x)

        
        return logits

    '''def forwardrandom(self, input_ids, segment_ids, input_mask):
        assert segment_ids.shape[1] == self.total_input_len
        zs = []
        select = randint(1, self.sequence_len - 1)
        for i in range(self.sequence_len):
            s_index = i*256
            t_index = s_index + 256
            cur_in    = input_ids[:,s_index:t_index] 
            cur_seg   = segment_ids[:,s_index:t_index]
            cur_mask  = input_mask[:,s_index:t_index]
            z = self.bert(cur_in, cur_seg, cur_mask, output_all_encoded_layers=False)[1]
            if i != select:
                z = z.detach()
            zs.append(z)

        cur_in    = input_ids[:,:256] 
        cur_seg   = segment_ids[:,:256]
        cur_mask  = input_mask[:,:256]
        first_z = self.bert(cur_in, cur_seg, cur_mask, output_all_encoded_layers=False)[1]

        zs = torch.stack(zs)

        x, _    = self.lstm(zs)
        x       = torch.cat([first_z, x[1], x[-1]], dim = 1)
        x       = self.dropout(x)
        logits  = self.classifier1(x)

        
        return logits



    def forward_noob_att(self, input_ids, segment_ids, input_mask):
        assert segment_ids.shape[1] == self.total_input_len
        zs = []
        if self.training:
            self.lstm.flatten_parameters()
            self.att.gru.flatten_parameters()

        for i in range(self.sequence_len):
            s_index = i*256
            t_index = s_index + 256
            cur_in    = input_ids[:,s_index:t_index] 
            cur_seg   = segment_ids[:,s_index:t_index]
            cur_mask  = input_mask[:,s_index:t_index]
            z = self.bert(cur_in, cur_seg, cur_mask, output_all_encoded_layers=False)[1].detach()
            zs.append(z)


        zs = torch.stack(zs)
        attention_output, norms = self.att(zs)
        x       = self.lstm(zs)[0]
        x       = x.view(self.sequence_len, -1, 2, 768*2)

        cur_in    = input_ids[:,:256] 
        cur_seg   = segment_ids[:,:256]
        cur_mask  = input_mask[:,:256]
        first_z = self.bert(cur_in, cur_seg, cur_mask, output_all_encoded_layers=False)[1]

        x       = torch.cat([first_z, x[0,:,1], x[1,:,0], x[-1,:,0],attention_output], dim = 1)
        x       = self.bn(x)
        x       = self.dropout(x)
        logits  = self.classifier1(x)

        
        return logits
    '''

    '''
        for i in range(self.sequence_len):
            s_index = i*256
            t_index = s_index + 256
            cur_in    = input_ids[:,s_index:t_index] 
            cur_seg   = segment_ids[:,s_index:t_index]
            cur_mask  = input_mask[:,s_index:t_index]
            z = self.bert(cur_in, cur_seg, cur_mask, output_all_encoded_layers=False)[1]
            #torch.index_select(z, 1, norm_index)
            for j in range(z.shape[0]):
                if i != norm_index[j]:
                    z[j].detach()#???
            zs2.append(z)
    
    '''

    '''
    def forward_bigatt(self, input_ids, segment_ids, input_mask):
        assert segment_ids.shape[1] == self.total_input_len
        zs = []
        if self.training:
            self.lstm.flatten_parameters()
            self.att.gru.flatten_parameters()

        input_ids   = input_ids.view(input_ids.size(0), self.sequence_len, -1)
        segment_ids = segment_ids.view(segment_ids.size(0), self.sequence_len, -1)
        input_mask  = input_mask.view(input_mask.size(0), self.sequence_len, -1)

        for i in range(self.sequence_len):
            z = self.bert(input_ids[:,i], segment_ids[:,i], input_mask[:,i], output_all_encoded_layers=False)[1].detach()
            zs.append(z)

        zs = torch.stack(zs)

        attention_output, norms = self.att(zs)
        #TODO: try argmax without first_z
        norm_index = torch.argmax(norms, dim = 1)

        att_id = [] # torch.zeros([input_ids.size(0),256]).cuda()
        att_seg = [] #torch.zeros([input_ids.size(0),256]).cuda()
        att_mask = [] #torch.zeros([input_ids.size(0),256]).cuda()

        for batch_index in range(input_ids.size(0)):
            i = norm_index[batch_index]
            att_id.append( input_ids[batch_index,i])
            att_seg.append(segment_ids[batch_index,i])
            att_mask.append(input_mask[batch_index,i])

        attention_z = self.bert(torch.stack(att_id), torch.stack(att_seg), torch.stack(att_mask), output_all_encoded_layers=False)[1]

        
        #pass norm index into fc_layers
        a = self.leaky(self.attention1(norms))
        a = self.leaky(self.attention2(a))
        #a = self.attention3(torch.cat([a,attention_z], dim = 1))
        a = self.attention3(torch.cat([a,attention_z], dim = 1))


        x       = self.lstm(zs)[0]
        #x       = x.view(self.sequence_len, -1, 2, 768*2)

        first_z = self.bert(input_ids[:,0], segment_ids[:,0], input_mask[:,0], output_all_encoded_layers=False)[1]

        #x       = torch.cat([first_z, x[0,:,1], x[1,:,0], x[-1,:,0],attention_output,a], dim = 1)
        x       = torch.cat([first_z, x[1], x[-1],attention_output,a], dim = 1)
        x       = self.leaky(self.classifier12_1(x))
       

        x       = self.dropout(x)
        logits  = self.classifier12_2(x)

        
        return logits
    
    def forward_diffh(self, input_ids, segment_ids, input_mask):
        assert segment_ids.shape[1] == self.total_input_len
        zs = []
        for i in range(self.sequence_len):
            s_index = i*256
            t_index = s_index + 256
            cur_in    = input_ids[:,s_index:t_index] 
            cur_seg   = segment_ids[:,s_index:t_index]
            cur_mask  = input_mask[:,s_index:t_index]
            z = self.bert(cur_in, cur_seg, cur_mask, output_all_encoded_layers=False)[1].detach()
            zs.append(z)

        cur_in    = input_ids[:,:256] 
        cur_seg   = segment_ids[:,:256]
        cur_mask  = input_mask[:,:256]
        _, first_z = self.bert(cur_in, cur_seg, cur_mask, output_all_encoded_layers=False)

        zs = torch.stack(zs)

        x, _    = self.lstm(zs)
        x       = torch.cat([first_z, x[0,:,768:], x[1,:,:768], x[-1]], dim = 1)
        x       = self.dropout(x)
        logits  = self.classifier1(x)

        
        return logits
    '''

class LongBert(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, sequence_len, input_len, num_labels=2):
        super(LongBert, self).__init__(config)
        self.num_labels = num_labels
        self.mem_size = config.hidden_size
        self.sequence_len = sequence_len
        self.total_input_len = sequence_len * input_len
        
        self.bert = BertModel(config)
        self.bert_layers = 12
        
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.lstm = nn.LSTM( config.hidden_size,  config.hidden_size * 2, bidirectional = True)
        self.classifier1 = torch.nn.Linear(config.hidden_size * 9, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids, input_mask):
        assert segment_ids.shape[1] == self.total_input_len
        if self.training:
            self.lstm.flatten_parameters()
        zs = []
        for i in range(self.sequence_len):
            s_index = i*256
            t_index = s_index + 256
            cur_in    = input_ids[:,s_index:t_index] 
            cur_seg   = segment_ids[:,s_index:t_index]
            cur_mask  = input_mask[:,s_index:t_index]
            z = self.bert(cur_in, cur_seg, cur_mask, output_all_encoded_layers=False)[1].detach()
            zs.append(z)

        cur_in    = input_ids[:,:256] 
        cur_seg   = segment_ids[:,:256]
        cur_mask  = input_mask[:,:256]
        _, first_z = self.bert(cur_in, cur_seg, cur_mask, output_all_encoded_layers=False)

        zs = torch.stack(zs)

        x, _    = self.lstm(zs)
        x       = torch.cat([first_z, x[1], x[-1]], dim = 1)
        x       = self.dropout(x)
        logits  = self.classifier1(x)

        
        return logits

class AblationLongBert(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, sequence_len, input_len, num_labels=2):
        super(AblationLongBert, self).__init__(config)
        self.num_labels = num_labels
        self.mem_size = config.hidden_size
        self.sequence_len = sequence_len
        self.total_input_len = sequence_len * input_len
        
        self.bert = BertModel(config)
        self.bert_layers = 12
        
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = torch.nn.Linear(config.hidden_size * sequence_len, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids, input_mask):
        assert segment_ids.shape[1] == self.total_input_len
        zs = []
        for i in range(self.sequence_len):
            s_index = i*256
            t_index = s_index + 256
            cur_in    = input_ids[:,s_index:t_index] 
            cur_seg   = segment_ids[:,s_index:t_index]
            cur_mask  = input_mask[:,s_index:t_index]
            z = self.bert(cur_in, cur_seg, cur_mask, output_all_encoded_layers=False)[1]
            if i > 0:
                z = z.detach()
            zs.append(z)

        x = torch.cat(zs, dim = 1)

        x       = self.dropout(x)
        logits  = self.classifier1(x)

        
        return logits