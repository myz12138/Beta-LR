from data_process import *
#c+q+o编码，c通过自编码器得到c的逻辑等价表示，然后进行数据增强
import math
from torch.distributions import beta
class Beta_LR(nn.Module):
    def __init__(self,pretrain_model,embedding_size,beta_size,out_size,project_layer_num,attention_layer_num,renew_layer_num,linear_layer_num,embed_method,operator_method) -> None:
        super().__init__()
        self.pretrain_model=pretrain_model
        self.embedding_size=embedding_size
        self.beta_size=beta_size#beta分布的维度
        self.out_size=out_size
        self.project_layer_num=project_layer_num#由点嵌入映射到beta嵌入的线性层数量-1
        self.attention_layer_num=attention_layer_num#intersection部分attention层数-1
        self.linear_layer_num=linear_layer_num#线性分类层数量-1
        self.renew_layer_num=renew_layer_num
        self.embed_method=embed_method#'point' or 'beta'
        self.operator_method=operator_method#'logic' or 'arithmetic'
        #initial network
        for i in range(project_layer_num):
            setattr(self,"projection_layer_%d"%i,nn.Linear(self.embedding_size, self.embedding_size).to(device))
            
        self.projection_layer=nn.Linear(self.embedding_size, 2*self.beta_size).to(device)
       # nn.init.xavier_uniform_(self.projection_layer.weight)

        self.projection_layer_b=nn.Linear(self.embedding_size, self.beta_size).to(device)
        nn.init.xavier_uniform_(self.projection_layer_b.weight)


        for i in range(attention_layer_num):
            setattr(self,'attention_layer_%d'%i,nn.Linear(2*self.beta_size, self.beta_size).to(device))
            nn.init.xavier_uniform_(self.attention_layer_0.weight)
        self.attention_layer=nn.Linear(self.beta_size, self.beta_size).to(device)
        nn.init.xavier_uniform_(self.attention_layer.weight)

        for i in range(linear_layer_num):
            setattr(self,'linear_layer_%d'%i,nn.Linear(8*self.beta_size,self.beta_size).to(device))
            nn.init.xavier_uniform_(self.linear_layer_0.weight)
        # self.linear_layer_mid=nn.Linear(4*self.beta_size, self.beta_size).to(device)
        # nn.init.xavier_uniform_(self.linear_layer_mid.weight)
        self.linear_layer=nn.Linear(self.beta_size, self.out_size).to(device)
        nn.init.xavier_uniform_(self.linear_layer.weight)

        
        self.loss_fn=nn.CrossEntropyLoss(reduction='mean').to(device)
    
    def init_embed(self,ids,mask,idx):#idx:8*4*2,获得初始嵌入
        o_embed,context_embed,q_embed,all_c_embed=list(),list(),list(),list()
        for i in range(len(ids)):
            init_embed_i=self.pretrain_model(**{'input_ids':ids[i],"attention_mask":mask[i]}).last_hidden_state#4*x*embedding_size
            o_embed_i,context_embed_i,q_embed_i,all_c_embed_i=list(),list(),list(),list()
            l_id=idx[i].shape[0]
            for j in range(len(init_embed_i)):
                c_mean_embed_i=[torch.mean(init_embed_i[j,1:idx[i][0,1].item(),:],dim=0)]
                if l_id>3:
                    for k in range(0,l_id-3):
                        c_mean_embed_i.append(torch.mean(init_embed_i[j,idx[i][k,1].item():idx[i][k+1,1].item(),:],dim=0))
                context_embed_i.append(torch.stack(c_mean_embed_i))#文本片段所有句子的嵌入,4*x*768
                q_embed_i.append(torch.mean(init_embed_i[j,idx[i][l_id-3,1].item():idx[i][l_id-2,1].item(),:],dim=0))
                o_embed_i.append(torch.mean(init_embed_i[j,idx[i][l_id-2,1].item():idx[i][l_id-1,1].item(),:],dim=0))#每个选项的答案片段
                all_c_embed_i.append(torch.mean(init_embed_i[j,1:idx[i][l_id-3,1].item(),:],dim=0))
            o_embed.append(torch.stack(o_embed_i))#8*4*768
            context_embed.append(torch.stack(context_embed_i))#8*4*x*768
            q_embed.append(torch.stack(q_embed_i))
            all_c_embed.append(torch.stack(all_c_embed_i))
        return torch.stack(o_embed),context_embed,torch.stack(q_embed),torch.stack(all_c_embed)#q，c，o的编码


    def get_param(self,embed):#将context的嵌入通过网络得到正例
        if self.embed_method=='beta':
            
            embed=1+F.relu(self.projection_layer(embed))
           
            a_embed,b_embed=torch.chunk(embed,2,dim=-1)
            #a_embed,b_embed=1+F.relu(self.projection_layer_a(embed)),1+F.relu(self.projection_layer_b(embed))
            return (a_embed,b_embed)
        else:
            embed=self.projection_layer(embed)
            return embed
        

    def get_union_embed(self,a_embed,b_embed):#得到四个选项的文本片段每个句子嵌入的并集,三维
        neg_embed_a,neg_embed_b=1/a_embed,1/b_embed#对每个句子都求反
        param_embed=torch.cat([neg_embed_a,neg_embed_b],dim=-1)#求反后求他们的交集
        for i in range(self.attention_layer_num):
            param_embed=F.relu(getattr(self,'attention_layer_%d' %i)(param_embed))
        param_embed=self.attention_layer(param_embed)
        w=torch.softmax(param_embed,dim=-2)
        neg_embed_a,neg_embed_b=w.mul(neg_embed_a),w.mul(neg_embed_b)#加权求和
        
        
        neg_embed_a,neg_embed_b=torch.sum(neg_embed_a,dim=-2),torch.sum(neg_embed_b,dim=-2)#加权求和,就是neg的交集，即所有句子反空间的交集
        union_embed_a,union_embed_b=1/neg_embed_a,1/neg_embed_b#所有句子空间的并
        
        return union_embed_a,union_embed_b

    
    def get_intersection_embed(self,a_embed,b_embed):#三维
        if self.operator_method=='logic':
            param_embed=torch.cat([a_embed,b_embed],dim=-1)
            for i in range(self.attention_layer_num):
                param_embed=F.relu(getattr(self,'attention_layer_%d' %i)(param_embed))
            param_embed=self.attention_layer(param_embed)
            w=torch.softmax(param_embed,dim=-2)
            intersection_embed_a,intersection_embed_b=w.mul(a_embed),w.mul(b_embed)
            intersection_embed_a,intersection_embed_b=torch.sum(intersection_embed_a,dim=-2),torch.sum(intersection_embed_b,dim=-2)
        else:#用平均法代替attention

            intersection_embed_a,intersection_embed_b=torch.mean(a_embed,dim=-2),torch.sum(b_embed,dim=-2)

        return intersection_embed_a,intersection_embed_b

    def get_renew_embed(self,a_embed_ci,b_embed_ci,inter_embed_a,inter_embed_b):
        
        a_embed_ci,b_embed_ci=a_embed_ci.unsqueeze(2),b_embed_ci.unsqueeze(2)#4*x*1*beta_size
        inter_embed_a,inter_embed_b=inter_embed_a.unsqueeze(1).repeat(1,a_embed_ci.shape[1],1),inter_embed_b.unsqueeze(1).repeat(1,b_embed_ci.shape[1],1)#4*x*beta_size
        cat_embed_a,cat_embed_b=torch.cat([a_embed_ci,inter_embed_a.unsqueeze(2)],dim=-2),torch.cat([b_embed_ci,inter_embed_b.unsqueeze(2)],dim=-2)#4*x*2*beta_size
        if self.operator_method=='logic':
            param_embed=torch.cat([cat_embed_a,cat_embed_b],dim=-1)
            for i in range(self.attention_layer_num):
                param_embed=F.relu(getattr(self,'attention_layer_%d' %i)(param_embed))
            param_embed=self.attention_layer(param_embed)
            w=torch.softmax(param_embed,dim=-2)
            new_a_embed_ci,new_b_embed_ci=w.mul(cat_embed_a),w.mul(cat_embed_b)
            new_a_embed_ci,new_b_embed_ci=torch.sum(new_a_embed_ci,dim=-2),torch.sum(new_b_embed_ci,dim=-2)
        else:
            new_a_embed_ci,new_b_embed_ci=torch.mean(cat_embed_a,dim=-2),torch.mean(cat_embed_b,dim=-2)
        return new_a_embed_ci,new_b_embed_ci
    
    def message_aggregation(self,context_embed,label):

        new_context_embed_a,new_context_embed_b=[],[]
        entropy_lis=[]
        for i in range(len(context_embed)):
            a_embed_ci,b_embed_ci=self.get_param(context_embed[i])[0],self.get_param(context_embed[i])[1]#4*x*embedding_size
            inter_embed_ai,inter_embed_bi=self.get_intersection_embed(a_embed_ci,b_embed_ci)#4*embeding_size
            new_a_embed_ci,new_b_embed_ci=self.get_renew_embed(a_embed_ci,b_embed_ci,inter_embed_ai,inter_embed_bi)#4*x*embedding_size

            entropy=self.get_kl(new_a_embed_ci,new_b_embed_ci,label[i])
            
            union_embed_a,union_embed_b=self.get_union_embed(new_a_embed_ci,new_b_embed_ci)#union
            #union_embed_a,union_embed_b=self.get_union_embed(a_embed_ci,b_embed_ci)#union
            new_context_embed_a.append(union_embed_a)#4*embeding_size
            new_context_embed_b.append(union_embed_b)#4*embeding_size
            entropy_lis.append(entropy)
        return torch.stack(new_context_embed_a),torch.stack(new_context_embed_b),entropy_lis#bacth*4*embeding_size
    
    def ablation_vector(self,context_embed,o_embed,q_embed,all_c_embed):#需要改attention和linear的layer大小，还有embed_method，保证维度为512
        union_embed,o_lis,q_lis,allc_lis=[],[],[],[]
        for i in range(len(context_embed)):
            embed_o=self.get_param(o_embed[i])#4*embeding_size
            embed_q=self.get_param(q_embed[i])
            embed_allc=self.get_param(all_c_embed[i])
            embed_c=self.get_param(context_embed[i])#4*x*embedding
            for j in range(self.attention_layer_num):
                param_embed=F.relu(getattr(self,'attention_layer_%d' %j)(embed_c))
            param_embed=self.attention_layer(param_embed)
            w=torch.softmax(param_embed,dim=-2)
            inter_embed_c=w.mul(embed_c)
            inter_embed_c=torch.sum(inter_embed_c,dim=-2)
            
            new_embed_c=torch.cat([embed_c.unsqueeze(2),inter_embed_c.unsqueeze(1).repeat(1,embed_c.shape[1],1).unsqueeze(2)],dim=-2)
            for j in range(self.attention_layer_num):
                param_embed=F.relu(getattr(self,'attention_layer_%d' %j)(new_embed_c))
            param_embed=self.attention_layer(param_embed)
            w=torch.softmax(param_embed,dim=-2)
            updated_embed_c=w.mul(new_embed_c)
            updated_embed_c=torch.sum(updated_embed_c,dim=-2)

            union_embed_c=torch.mean(updated_embed_c,dim=-2)
            union_embed.append(union_embed_c)
            o_lis.append(embed_o)
            q_lis.append(embed_q)
            allc_lis.append(embed_allc)
        union_embed,new_o,new_q,new_allc=torch.stack(union_embed),torch.stack(o_lis),torch.stack(q_lis),torch.stack(allc_lis)
        cat_embed=torch.cat((union_embed,new_allc,new_o,new_q),dim=-1)
        for i in range(self.linear_layer_num):
            cat_embed=F.relu(getattr(self,'linear_layer_%d'%i)(cat_embed))
        
        output=self.linear_layer(cat_embed)
        return output


    def classify(self,context_embed,o_embed,q_embed,all_c_embed,label):#计算最基本的loss，对每个选项都算
        a_embed_o,b_embed_o=self.get_param(o_embed)[0],self.get_param(o_embed)[1]#bacth*4*embeding_size
        a_embed_q,b_embed_q=self.get_param(q_embed)[0],self.get_param(q_embed)[1]
        a_embed_allc,b_embed_allc=self.get_param(all_c_embed)[0],self.get_param(all_c_embed)[1]

        beta_embed_o,beta_embed_q,beta_embed_allc=torch.cat([a_embed_o,b_embed_o],dim=-1),torch.cat([a_embed_q,b_embed_q],dim=-1),torch.cat([a_embed_allc,b_embed_allc],dim=-1)
        new_context_embed_a,new_context_embed_b,entropy_tensor=self.message_aggregation(context_embed,label)#聚合所有子句信息后，加权得到的新的embed，并将所有embed求并集
        
        new_context_embed=torch.cat([new_context_embed_a,new_context_embed_b],dim=-1)
        out_embed=torch.cat((new_context_embed,beta_embed_allc,beta_embed_o,beta_embed_q),dim=-1)
        for i in range(self.linear_layer_num):
            out_embed=F.relu(getattr(self,'linear_layer_%d'%i)(out_embed))
        
        output=self.linear_layer(out_embed)
        return output,entropy_tensor
        
    def get_kl(self,a_embed_c,b_embed_c,label_i):
        m=beta.Beta(a_embed_c,b_embed_c)
        
        #a=torch.distributions.kl.kl_divergence(m,n)
        entropy=m.entropy()
        entropy=-torch.mean(entropy,dim=-1)[label_i]
        # mean = torch.mean(entropy)
        # std = torch.std(entropy)
        # z_score_normalized_tensor = (entropy - mean) / std
        return entropy#z_score_normalized_tensor 

    def forward(self,batch,flag):
        ids,mask,idx,label=batch[0],batch[1],batch[2],batch[3]
        o_embed,context_embed,q_embed,all_c_embed=self.init_embed(ids,mask,idx)
        output,entropy_tensor=self.classify(context_embed,o_embed,q_embed,all_c_embed,label)
        #output=self.ablation_vector(context_embed,o_embed,q_embed,all_c_embed)
        output=output.squeeze(2)
        loss=self.loss_fn(output,torch.tensor(label).to(device))
    
        if flag=='train':
            return loss,entropy_tensor#+0.01*loss_regular
        else:
            return output,loss,entropy_tensor#+0.01*loss_regular
            

