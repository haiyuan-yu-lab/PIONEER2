import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np

class Transformer_net(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout):
        super(Transformer_net, self).__init__()
        
        self.fc = nn.Linear(d_model, dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.fc(x)
        x = self.transformer_encoder(x)
        return x
    
class CNNConvLayersPre(nn.Module):
    def __init__(self, kernels):
        super(CNNConvLayersPre,self).__init__()
        
        self.cnn_conv = nn.ModuleList()
        for l in range(len(kernels)):
            self.cnn_conv.append(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernels[l],1), padding=(kernels[l]//2,0)))
            self.cnn_conv.append(nn.PReLU())

    def forward(self, x):
        for l , m in enumerate(self.cnn_conv):
            x = m(x)
        return x
    
class Self_Attention(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads):
        super(Self_Attention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.output_dim)
        self.fc2 = nn.Linear(self.output_dim, self.n_heads)

    def forward(self, h):
        x = torch.tanh(self.fc1(h))
        x = self.fc2(x)
        x = x.transpose(0, 1)
        attention = torch.softmax(x, dim=1)
        
        x = torch.mm(attention, h)
        x = torch.sum(x, 0, keepdim=True)/self.n_heads
        return x
    
class GMM(nn.Module):
    def __init__(self, in_dim_gmm, num_heads):
        super(GMM, self).__init__()
        self.SAP = Self_Attention(in_dim_gmm, in_dim_gmm//2, num_heads)
        
    def forward(self, aa_gmms, atom_gmms, atom_nums):
        aa_gmms = aa_gmms.float()
        atom_gmms = atom_gmms.float()
        
        aa_cnt, atom_cnt = 0, 0
        for i in range(len(atom_nums)):
            for j in range(len(atom_nums[i])):
                aa_cnt += atom_nums[i][j].shape[0]
                atom_cnt += atom_nums[i][j][-1][-1]+1
        assert aa_cnt == aa_gmms.size()[0]
        assert atom_cnt == atom_gmms.size()[0]
        
        atom_nums = [j for i in atom_nums for j in i]
        ind = atom_nums[0][-1][-1]+1
        for i in range(1, len(atom_nums)):
            atom_nums[i] += ind
            ind = atom_nums[i][-1][-1]+1
        atom_nums = np.concatenate(atom_nums)
            
        atom_gmms_ = []
        for i in range(atom_nums.shape[0]):
            atom_gmms_.append(self.SAP(atom_gmms[atom_nums[i][0]:atom_nums[i][1]+1, :]))
        atom_gmms_ = torch.cat(atom_gmms_, 0)
        assert atom_nums[i][1]+1 == atom_gmms.size()[0]
        return torch.cat((aa_gmms, atom_gmms_), 1)
    
class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=2):
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        y = torch.sigmoid(y)
        return y
    
class Network(nn.Module):

    def __init__(self, config):
        super().__init__()

        net_params = config['net_params']
        self.device = config['device']
        use_edge = net_params['use_edge']
        in_dim_node = net_params['in_dim_node']
        in_dim_edge = net_params['in_dim_edge']
        in_dim_gmm = net_params['in_dim_gmm']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        layer_norm = net_params['layer_norm']
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = net_params['max_wl_role_index']
        self.use_weight_in_loss = net_params['use_weight_in_loss']
        kernels = net_params['kernels']
        
        num_heads_trans = net_params['num_heads_trans']
        hidden_dim_trans = net_params['hidden_dim_trans']
        n_layers_trans = net_params['n_layers_trans']
        dropout_trans = net_params['dropout_trans']
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        if use_edge:
            from gt_edge_layer import GraphTransformerLayer
            self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        else:
            from gt_layer import GraphTransformerLayer
        
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, 
                                                           dropout, layer_norm, batch_norm, residual) for _ in range(n_layers)])
        
        self.transformer_net = Transformer_net(in_dim_node, num_heads_trans, hidden_dim_trans, n_layers_trans, dropout_trans)
        self.cnn_conv_pre = CNNConvLayersPre(kernels)
        
        self.gmm = GMM(in_dim_gmm, num_heads)
        
        self.SAP1 = Self_Attention(hidden_dim, hidden_dim//2, num_heads)
        self.SAP2 = Self_Attention(hidden_dim_trans, hidden_dim_trans//2, num_heads)
        
        self.MLP_layer = MLPReadout(hidden_dim+hidden_dim+hidden_dim_trans+hidden_dim_trans, n_classes)

    def forward(self, target_g, target_aa_gmms, target_atom_gmms, target_atom_nums, target_h_lap_pos_enc, target_h_wl_pos_enc, partner_g, partner_aa_gmms, partner_atom_gmms, partner_atom_nums, partner_h_lap_pos_enc, partner_h_wl_pos_enc, batch_nums):
        target_aa_gmms_embedding = self.gmm(target_aa_gmms, target_atom_gmms, target_atom_nums)
        partner_aa_gmms_embedding = self.gmm(partner_aa_gmms, partner_atom_gmms, partner_atom_nums)
        
        target_embedding_h_ori = torch.cat((target_g.ndata['feat'].float(), target_aa_gmms_embedding), 1)
        partner_embedding_h_ori = torch.cat((partner_g.ndata['feat'].float(), partner_aa_gmms_embedding), 1)
        
        target_embedding1_h = self.embedding_h(target_embedding_h_ori)
        target_embedding1_e = self.embedding_e(target_g.edata['feat'].float())
        partner_embedding1_h = self.embedding_h(partner_embedding_h_ori)
        partner_embedding1_e = self.embedding_e(partner_g.edata['feat'].float())
        
        if self.lap_pos_enc:
            target_h_lap_pos_enc = self.embedding_lap_pos_enc(target_h_lap_pos_enc.float())
            target_embedding1_h = target_embedding1_h + target_h_lap_pos_enc
        if self.wl_pos_enc:
            target_h_wl_pos_enc = self.embedding_wl_pos_enc(target_h_wl_pos_enc)
            target_embedding1_h = target_embedding1_h + target_h_wl_pos_enc
        target_embedding1_h = self.in_feat_dropout(target_embedding1_h)
        
        for conv in self.layers:
            target_embedding1_h, target_embedding1_e = conv(target_g, target_embedding1_h, target_embedding1_e)
            
        if self.lap_pos_enc:
            partner_h_lap_pos_enc = self.embedding_lap_pos_enc(partner_h_lap_pos_enc.float())
            partner_embedding1_h = partner_embedding1_h + partner_h_lap_pos_enc
        if self.wl_pos_enc:
            partner_h_wl_pos_enc = self.embedding_wl_pos_enc(partner_h_wl_pos_enc)
            partner_embedding1_h = partner_embedding1_h + partner_h_wl_pos_enc
        partner_embedding1_h = self.in_feat_dropout(partner_embedding1_h)
        
        for conv in self.layers:
            partner_embedding1_h, partner_embedding1_e = conv(partner_g, partner_embedding1_h, partner_embedding1_e)
            
        batch_nums = np.concatenate(batch_nums)
        for i in range(1, len(batch_nums)):
            cnt1 = batch_nums[i][1]-batch_nums[i][0]+1
            cnt2 = batch_nums[i][3]-batch_nums[i][2]+1
            batch_nums[i][0] = batch_nums[i-1][1]+1
            batch_nums[i][1] = batch_nums[i-1][1]+cnt1
            batch_nums[i][2] = batch_nums[i-1][3]+1
            batch_nums[i][3] = batch_nums[i-1][3]+cnt2
            
        assert batch_nums[-1][3]+1 == partner_embedding1_h.size()[0]
            
        partner_embedding1_h_ = []
        for i in range(batch_nums.shape[0]):
            partner_embedding1_h_.append(self.SAP1(partner_embedding1_h[batch_nums[i][2]:batch_nums[i][3]+1, :]).expand(batch_nums[i][1]-batch_nums[i][0]+1, -1))
        partner_embedding1_h_ = torch.cat(partner_embedding1_h_, 0)
        
        assert batch_nums[-1][1]+1 == target_embedding_h_ori.size()[0]
        assert batch_nums[-1][3]+1 == partner_embedding_h_ori.size()[0]
        
        target_embedding2_h = []
        for i in range(batch_nums.shape[0]):
            tmp = self.transformer_net(target_embedding_h_ori[batch_nums[i][0]:batch_nums[i][1]+1, :][np.newaxis,:,:])
            tmp = torch.squeeze(tmp, 0)
            target_embedding2_h.append(tmp)
        target_embedding2_h = torch.cat(target_embedding2_h, 0)
        partner_embedding2_h = []
        for i in range(batch_nums.shape[0]):
            tmp = self.transformer_net(partner_embedding_h_ori[batch_nums[i][2]:batch_nums[i][3]+1, :][np.newaxis,:,:])
            tmp = torch.squeeze(tmp, 0)
            partner_embedding2_h.append(tmp)
        partner_embedding2_h = torch.cat(partner_embedding2_h, 0)
        
        target_embedding2_h_ = []
        for i in range(batch_nums.shape[0]):
            tmp = self.cnn_conv_pre(target_embedding2_h[batch_nums[i][0]:batch_nums[i][1]+1, :][np.newaxis,np.newaxis,:,:])
            tmp = torch.squeeze(tmp, 0)
            tmp = torch.squeeze(tmp, 0)
            target_embedding2_h_.append(tmp)
        target_embedding2_h_ = torch.cat(target_embedding2_h_, 0)
        partner_embedding2_h_ = []
        for i in range(batch_nums.shape[0]):
            tmp = self.cnn_conv_pre(partner_embedding2_h[batch_nums[i][2]:batch_nums[i][3]+1, :][np.newaxis,np.newaxis,:,:])
            tmp = torch.squeeze(tmp, 0)
            tmp = torch.squeeze(tmp, 0)
            partner_embedding2_h_.append(tmp)
        partner_embedding2_h_ = torch.cat(partner_embedding2_h_, 0)
        
        partner_embedding2_h__ = []
        for i in range(batch_nums.shape[0]):
            partner_embedding2_h__.append(self.SAP2(partner_embedding2_h_[batch_nums[i][2]:batch_nums[i][3]+1, :]).expand(batch_nums[i][1]-batch_nums[i][0]+1, -1))
        partner_embedding2_h__ = torch.cat(partner_embedding2_h__, 0)
            
        h = torch.cat((target_embedding1_h, partner_embedding1_h_, target_embedding2_h_, partner_embedding2_h__), 1)
        h_out = self.MLP_layer(h)
        return h_out
    
    def loss(self, pred, label):
        if self.use_weight_in_loss:
            V = label.size(0)
            label_count = torch.bincount(label.long())
            cluster_sizes = torch.zeros(label_count.size(0)).long().to(self.device)
            cluster_sizes[torch.arange(label_count.size(0)).long()] = label_count
            weight = (V - cluster_sizes).float() / V
            weight *= (cluster_sizes>0).float()
            
            criterion = nn.BCELoss(weight=weight[label.long()])
        else:
            criterion = nn.BCELoss()
            
        loss = criterion(pred, label)
        return loss
    