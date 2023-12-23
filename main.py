#trans 1.12.1+cuda 10.2
from data_process import *
from Beta-LR_model import *
import torch
from utils import *
def test(test_dataloader,model):
    model.eval()
    test_label=[]
    test_sum=0
    for step, test_batch in enumerate(test_dataloader):
        output,loss,entropy=model(test_batch)
        pred=torch.argmax(output,dim=1)
        test_label+=[pred[i].item() for i in range(len(pred))]
        test_=torch.eq(pred,torch.tensor(test_batch[3]).to(device))
        test_sum+=test_.sum().item()
    return  test_label,test_sum/test_length

def train(train_dataloader,model, optimizer):
    model.train()
    train_sum=0
    train_loss_num=0
    for step, train_batch in enumerate(train_dataloader):
        output,train_loss,entropy=model(train_batch)
        pred=torch.argmax(output,dim=1)
        train_=torch.eq(pred,torch.tensor(train_batch[3]).to(device))
        train_sum+=train_.sum().item()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_loss_num+=train_loss.item()*len(train_batch[0])
        if step%20==0:    
            print(f"loss:  {train_loss.item():>7f} [{step:>5d}/{len(train_dataloader):>5d}]")
    return train_sum/train_length,train_loss_num/train_length
def val(val_dataloader,model):
    model.eval()
    kl_sum=0
    val_sum,val_loss_sum=0,0
    for step, val_batch in enumerate(val_dataloader):
        output,val_loss,entropy=model(val_batch)
        pred=torch.argmax(output,dim=1)
        val_=torch.eq(pred,torch.tensor(val_batch[3]).to(device))
        val_sum+=val_.sum().item()
        val_loss_sum+=val_loss.item()*len(val_batch[0])
        if step%10==0:
            print(f"loss:  [{step:>5d}/{len(val_dataloader):>5d}]")
    return  val_sum/val_length,val_loss_sum/val_length


def get_dataset(option):
    if args.data_name=='ReClor':
        if option=='train':
            context_list,answers_list,question_list,label_list=read_ReClor_data(option='train')
        elif option=='val':
            context_list,answers_list,question_list,label_list=read_ReClor_data(option='val')
        elif option=='test':
            context_list,answers_list,question_list,label_list=read_ReClor_data('test')
        else:
            print('error')
    else:
        if option=='train':
            context_list,answers_list,question_list,label_list=read_LogiQA_data(option='train')
        elif option=='val':
            context_list,answers_list,question_list,label_list=read_LogiQA_data(option='val')
        elif option=='test':
            context_list,answers_list,question_list,label_list=read_LogiQA_data('test')
        else:
            print('error')
    texts=sentence_data(context_list,answers_list,question_list)
    ids,mask,idx=encode_sentence_texts(texts)
    dataset=data_Loader(ids,mask,idx,label_list)
    return dataset

if __name__=="__main__":
    train_dataset=get_dataset('train')
    train_Dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_batch)

    val_dataset=get_dataset('val') 
    val_Dataloader=DataLoader(val_dataset,batch_size=1,shuffle=False,collate_fn=collate_batch)

    test_dataset=get_dataset('test')
    test_Dataloader=DataLoader(test_dataset,batch_size=1,shuffle=False,collate_fn=collate_batch)
    
    train_length,val_length,test_length=len(train_dataset),len(val_dataset),len(test_dataset)
    model=Beta_LR(
                        pretrain_model=pretrain_model,
                        embedding_size=args.embedding_size,
                        beta_size=args.beta_size,
                        out_size=1,
                        attention_layer_num=args.attention_layer_num,
                        embed_method=args.embed_method,
                        operator_method=args.operator_method
                        
                    )
    optimizer=torch.optim.AdamW(params=model.parameters(),lr=args.lr,betas=[args.B1,args.B2])
    train_lis,val_lis,test_lis=[],[],[]
    print("---------------------start model3  training ------------------------------------------------")
    best_acc=0
    for epoch_i in range(args.epoch):
        train_acc,train_loss=train(train_Dataloader,model,optimizer)
        val_acc,val_loss=val(val_Dataloader,model)
        train_lis.append(train_acc)
        val_lis.append(val_acc)
        print("val_acc:",val_acc)
        print('Epoch3: {0}, Train loss3: {1} , val loss3: {2},'.format(epoch_i, train_loss,val_loss))
        print("train_lis:{0},val_lis:{1}".format(train_lis,val_lis))
        if val_acc>best_acc:
            torch.save(model.state_dict(),model_path)
            best_loss=val_acc
            
    print("---------------------start  test------------------------------------------------")
    model.load_state_dict(torch.load(model_path))
    test_label,test_acc=test(test_Dataloader,model)
    if args.data_name=='LogiQA':
        print("The test acc of LogiQA is:",test_acc)
    else:
        np.save(label_path,test_label)
        print("The test label of ReClor have been saved in label_path")