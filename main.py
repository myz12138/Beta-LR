#1.12.1+cu102
from data_process import *
from kl_model import *
import torch
import csv
def test(test_dataloader,model):
    model.eval()
    test_label=[]
    test_sum=0
    for step, test_batch in enumerate(test_dataloader):
        output,loss,kl_i=model(test_batch,'val')
        pred=torch.argmax(output,dim=1)
        test_label+=[pred[i].item() for i in range(len(pred))]
        test_=torch.eq(pred,torch.tensor(test_batch[3]).to(device))
        test_sum+=test_.sum().item()
    return  test_label,test_sum/1000

def train(train_dataloader,model, optimizer):
    model.train()
    train_sum=0
    train_loss_num=0
    kl_sum=0
    for step, train_batch in enumerate(train_dataloader):
        output,train_loss,kl_i=model(train_batch,'val')
        pred=torch.argmax(output,dim=1)
        train_=torch.eq(pred,torch.tensor(train_batch[3]).to(device))
        train_sum+=train_.sum().item()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_loss_num+=train_loss.item()*len(train_batch[0])
        if step%20==0:    
            print(f"loss:  {train_loss.item():>7f} [{step:>5d}/{len(train_dataloader):>5d}]")
    return train_sum,train_loss_num/4638,kl_sum/4638#4638

def val(val_dataloader,model):
    model.eval()
    kl_sum=0
    val_sum,val_loss_sum=0,0
    entropy_list=[]
    for step, val_batch in enumerate(val_dataloader):
        output,val_loss,entropy_i=model(val_batch,'val')
        entropy_list.append(entropy_i[0].tolist())
        pred=torch.argmax(output,dim=1)
        val_=torch.eq(pred,torch.tensor(val_batch[3]).to(device))
        val_sum+=val_.sum().item()

        val_loss_sum+=val_loss.item()*len(val_batch[0])
        if step%10==0:
            print(f"loss:  [{step:>5d}/{len(val_dataloader):>5d}]")
    return  val_sum,val_loss_sum/500,kl_sum/500,entropy_list


def collate_batch3(data):
        unit_ids,unit_mask,unit_idx,unit_label=[],[],[],[]
        for unit in data:
            unit_ids.append(unit[0])
            unit_mask.append(unit[1])
            unit_idx.append(unit[2])
            unit_label.append(unit[3])
        return  unit_ids,unit_mask,unit_idx,unit_label

def get_dataset(path,option):
    if option=='train':#train
        context_list,answers_list,question_list,label_list=read_json_data(path,option='train')
    elif option=='val':#val
        context_list,answers_list,question_list,label_list=read_json_data(path,option='val')
    elif option=='test':
        context_list,answers_list,question_list,label_list=read_json_data(path,'test')
    else:
        print('error')
    texts3=sentence_data(context_list,answers_list,question_list)
    ids3,mask3,idx3=encode_sentence_texts(texts3)
    dataset3=data_Loader3(ids3,mask3,idx3,label_list)
    return dataset3#,dataset3
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__=="__main__":
    #loss_fn=torch.nn.CrossEntropyLoss().to(device)
    
# 设置随机数种子
    #setup_seed(3407)
    path='./Beta-LR1'
    loss_path='./Beta-LR1/loss.npy' 
    label_path='./Beta-LR1/label.npy' 

    MODEL_PATH3="./Beta-LR1/net_params_sentence.pkl" #net3存的是只有pos，endtoend存的是对比
    #label_path='./base_logic_extension/label.npy' 
    
    train_dataset3=get_dataset(path,'train')
    # #train_Dataloader1=DataLoader(train_dataset1,batch_size=16,shuffle=True)
    train_Dataloader3=DataLoader(train_dataset3,batch_size=8,shuffle=True,collate_fn=collate_batch3)

    val_dataset3=get_dataset(path,'val') 
    val_Dataloader3=DataLoader(val_dataset3,batch_size=1,shuffle=False,collate_fn=collate_batch3)
    #val_Dataloader1=DataLoader(val_dataset1,batch_size=8,shuffle=False)
    
    test_dataset3=get_dataset(path,'test')
    test_Dataloader3=DataLoader(test_dataset3,batch_size=8,shuffle=False,collate_fn=collate_batch3)
    #test_Dataloader1=DataLoader(test_dataset1,batch_size=8,shuffle=False,collate_fn=collate_batch1)
    
    best_loss3=0
    epoch3=20
    
    print("---------------------start model3  training ------------------------------------------------")
    model3=Beta_LR(
                        pretrain_model=pretrain_model,
                        embedding_size=1024,
                        beta_size=128,
                        out_size=1,
                        project_layer_num=0,
                        attention_layer_num=1,
                        renew_layer_num=1,
                        linear_layer_num=1,
                        embed_method='beta',
                        operator_method='logic'
                        
                    )
    optimizer3=torch.optim.AdamW(params=model3.parameters(),lr=1e-6,betas=[0.9,0.99])
    val_lis=[]
    train_lis=[]
    kl_val_lis,kl_train_lis=[],[]
    for e3 in range(epoch3):
        train_sum,train_loss3,kl_train_sum=train(train_Dataloader3,model3,optimizer3)
        val_sum,val_loss3,kl_val_sum,entropy_list=val(val_Dataloader3,model3)
        # with open('./Beta-LR1/entropy.csv', 'w') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(entropy_list)
        train_lis.append(train_sum/len(train_dataset3))
        val_lis.append(val_sum/len(val_dataset3))
        kl_train_lis.append(kl_train_sum)
        kl_val_lis.append(kl_val_sum)
        print("val_acc:",val_sum/len(val_dataset3))
        print('Epoch3: {0}, Train loss3: {1} , val loss3: {2},'.format(e3, train_loss3,val_loss3))
        print("train_lis:{0},val_lis:{1},kl_train{2},kl_val:{3}".format(train_lis,val_lis,kl_train_lis,kl_val_lis))
        if val_sum>best_loss3:
            torch.save(model3.state_dict(),MODEL_PATH3)
            best_loss3=val_sum
            
            
        
        
    print("---------------------start  test------------------------------------------------")
    model3.load_state_dict(torch.load(MODEL_PATH3))
    test_label,test_sum=test(test_Dataloader3,model3)
    print(test_sum)
    np.save(label_path,test_label)
    data = np.load('./Beta-LR1/label.npy',allow_pickle=True)

# 打印读取的数据
    print(data)
