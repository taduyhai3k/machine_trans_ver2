import torch
import os
import csv
import numpy as np
import nltk
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def write_accuracy_to_csv(filename, train, valid, test):
    fieldnames = ["Train Accuracy", "Validation Accuracy", "Test Accuracy", "Train Loss", "Validation Loss", "Test Loss"]
    fieldnames_pararel = ["Train Accuracy E2V", "Train Accuracy V2E", "Validation Accuracy E2V", "Validation Accuracy V2E", "Test Accuracy E2V", "Test Accuracy V2E", "Train Loss E2V", "Train Loss V2E", "Validation Loss E2V", "Validation Loss V2E", "Test Loss E2V", "Test Loss V2E"]
    
    folder, file_name = os.path.split(filename)
    
    # Kiểm tra xem file đã tồn tại chưa
    if not os.path.isfile(filename):
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Nếu file chưa tồn tại, tạo một file mới và ghi thông tin đầu tiên vào nó
        with open(filename, mode='x', newline='') as file:
            if len(train) < 4:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({"Train Accuracy": train[1], "Validation Accuracy": valid[1], "Test Accuracy": test[1], "Train Loss": train[0], "Validation Loss": valid[0], "Test Loss": test[0]})
            else:
                writer = csv.DictWriter(file, fieldnames=fieldnames_pararel)
                writer.writeheader()
                writer.writerow({
                    "Train Accuracy E2V": train[1], "Train Accuracy V2E": train[3],
                    "Validation Accuracy E2V": valid[1], "Validation Accuracy V2E": valid[3],
                    "Test Accuracy E2V": test[1], "Test Accuracy V2E": test[3],
                    "Train Loss E2V": train[0], "Train Loss V2E": train[2],
                    "Validation Loss E2V": valid[0], "Validation Loss V2E": valid[2],
                    "Test Loss E2V": test[0], "Test Loss V2E": test[2]
                })
    else:
        # Nếu file đã tồn tại, mở file trong chế độ append để thêm dữ liệu mới vào cuối file
        with open(filename, mode='a', newline='') as file:
            if len(train) < 4:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow({
                    "Train Accuracy": train[1], "Validation Accuracy": valid[1], "Test Accuracy": test[1],
                    "Train Loss": train[0], "Validation Loss": valid[0], "Test Loss": test[0]
                })
            else:
                writer = csv.DictWriter(file, fieldnames=fieldnames_pararel)
                writer.writerow({
                    "Train Accuracy E2V": train[1], "Train Accuracy V2E": train[3],
                    "Validation Accuracy E2V": valid[1], "Validation Accuracy V2E": valid[3],
                    "Test Accuracy E2V": test[1], "Test Accuracy V2E": test[3],
                    "Train Loss E2V": train[0], "Train Loss V2E": train[2],
                    "Validation Loss E2V": valid[0], "Validation Loss V2E": valid[2],
                    "Test Loss E2V": test[0], "Test Loss V2E": test[2]
                })

def SparseCrossEntropy(true, pred):
    #shape pred is [batch_size, length, embed_size]
    #shape true is [batch_size, length]
    pred_tmp = torch.gather(pred.softmax(dim = -1)[:, :-1, :], dim = -1, index = true.unsqueeze(dim = -1)[:, 1:,:])
    pred_tmp[pred_tmp < 1e-40] = 1e-40 
    weights = true.unsqueeze(dim = -1)[:, 1:,:] > 0
    return (torch.log(pred_tmp) * -1 * weights).sum() / (weights.sum())

def transformer_lr(step_num, d_model = 512, warmup_steps = 4000, max_step = 40000):  
    if max_step > warmup_steps:  
        step_tmp = step_num % max_step           
    if step_tmp == 0:
        step_tmp += 1
    if step_num > max_step:
        step_tmp += warmup_steps / 2     
    lr = d_model ** (-0.5) * min(step_tmp ** (-0.5), step_tmp * warmup_steps ** (-1.5))
    return lr

def bleu_score(infer, candi):
    candi_numpy = candi.to('cpu').detach().numpy().astype(str).tolist()
    infer_numpy = infer.to('cpu').detach().numpy().astype(str).tolist()
    return nltk.translate.bleu_score.corpus_bleu(infer_numpy,candi_numpy)

def accuracy(true, pred):
    #shape pred is [batch_size, length, vocab_size]
    #shape true is [batch_size, length]    
    pred_tmp = torch.argmax(pred, dim = -1)[:, :-1]
    weights = true[:, 1:] > 0
    return ((pred_tmp == true[:,1:]) * weights).sum() / (weights.sum())

def predict(model, data, inp_tokennizer, out_tokenizer, max_lenth = 300):
    #translate one sentences
    model.eval()
    inp_encode = inp_tokennizer.encode(data).reshape(1, -1)
    out_encode = out_tokenizer.encode('<start>')[1].reshape(1, -1)
    with torch.no_grad():
        for i in range(max_lenth):
            pred = model(inp_encode, out_encode)
            preds = pred[:, -1, :].reshape(1, -1)
            predict_id = torch.argmax(preds, dim = -1, keepdim= True).type(torch.int64)
            out_encode = torch.cat([out_encode, predict_id], dim = -1)
            if out_tokenizer.encode('<end>')[1] == predict_id:
                return out_tokenizer.decode(out_encode.reshape(-1))
    return out_tokenizer.decode(out_encode.reshape(-1))     

def predict_parallel(model, data, inp_tokennizer, out_tokenizer, inp = 'E',max_lenth = 300):
    #translate one sentences
    model.eval()
    inp_encode = inp_tokennizer.encode(data).reshape(1, -1)
    out_encode = out_tokenizer.encode('<start>')[1].reshape(1, -1)
    with torch.no_grad():
        for i in range(max_lenth):
            pred = model(inp_encode, out_encode, inp)
            preds = pred[:, -1, :].reshape(1, -1)
            predict_id = torch.argmax(preds, dim = -1, keepdim= True).type(torch.int64)
            out_encode = torch.cat([out_encode, predict_id], dim = -1)
            if out_tokenizer.encode('<end>')[1] == predict_id:
                return out_tokenizer.decode(out_encode.reshape(-1))
    return out_tokenizer.decode(out_encode.reshape(-1))     

def save(path, model, optimizer, scheduler, epoch, score):
    state = {
        'model' : model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'score': score
    }           
    torch.save(state, path)
    print("save successful")

def load(path, model, optimizer, scheduler):
    if not torch.cuda.is_available():
        state = torch.load(path, map_location=torch.device('cpu'))
    else:
        state = torch.load(path)    
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scheduler.optimizer = optimizer
    epoch = state['epoch']
    score = state['score']
    print("load successful")
    return model, optimizer, scheduler, epoch, score            

def eval(model, data_loader,optimizer, scheduler, is_training = True):
    mean_loss = []      
    acc = []  
    infer = None
    candidate = None    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    if is_training:
        model.train()
        data_iter = tqdm(data_loader, desc='Training', position=0, leave=True)     
        i = 1   
        for input, target in data_iter:
            optimizer.zero_grad()
            input, target = input.to(device), target.to(device)
            output = model(input, target)
            loss = SparseCrossEntropy(target, output)           
            mean_loss.append(loss.item())
            acc.append(accuracy(target, output).item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 200 == 0:
                print(f"Vòng lặp thứ {i}, loss {np.mean(np.array(mean_loss))}, acc {np.mean(np.array(acc))}")    
            i += 1       
            #if infer is not None:
                #infer = torch.cat([infer, output.to('cpu').argmax(dim = -1)], dim = 0)
            #else:
                #infer = output.to('cpu').argmax(dim = -1)
            #if candidate is not None:
                #candidate = torch.cat([candidate, target.to('cpu')], dim = 0)   
            #else:
                #candidate = target.to('cpu')                 
    else:
        model.eval()    
        with torch.no_grad():
            mean_loss = []   
            acc = []     
            infer = None
            candidate = None
            data_iter = tqdm(data_loader, desc='Not training', position=0, leave=True)        
            for input, target in data_iter:
                input, target = input.to(device), target.to(device)
                output = model(input, target)
                loss  = SparseCrossEntropy(target, output)
                mean_loss.append(loss.item())
                acc.append(accuracy(target, output).item())
                #if infer is not None:
                    #infer = torch.cat([infer, output], dim = 0)
                #else:
                    #infer = output
                #if candidate is not None:
                    #candidate = torch.cat([candidate, target], dim = 0)   
                #else:
                    #candidate = target      
    return np.mean(np.array(mean_loss)), np.mean(np.array(acc))            
                
                    

def train(model, optimizer, epoch, datatrain_loader,datavalid_loader = None, datatest_loader = None,path = "", path1 = ""):
    scheduler = LambdaLR(optimizer, lr_lambda= lambda step_num : transformer_lr(step_num=step_num, d_model=model.dmodel))
    tmp_score = 1e30
    if os.path.isfile(path):
        model, optimizer, scheduler, epoch_old, tmp_score = load(path, model, optimizer, scheduler)
    else:
        epoch_old = 0    
    for i in tqdm(range(epoch_old, epoch), desc='Epoch', position=0, leave=True):
        result_train =  eval(model, datatrain_loader, optimizer, scheduler, True)   
        if datavalid_loader is not None:
            result_valid = eval(model, datavalid_loader, optimizer, scheduler, False)
        else:
            result_valid = [0,0]
        if datatest_loader is not None:
            result_test = eval(model, datatest_loader, optimizer,scheduler, False)
        else:
            result_test = [0,0]            
        print(f'\n Loss train {result_train[0]}, Acc train {result_train[1]};Loss valid {result_valid[0]}, Acc valid {result_valid[1]};Loss test {result_test[0]}, Acc test {result_test[1]}.')    
        if result_train[0] + result_valid[0] + result_test [0] < tmp_score:
            if path is not None:
                save(path, model, optimizer, scheduler, i, result_train[0] + result_valid[0] + result_test [0])
                tmp_score = result_train[0] + result_valid[0] + result_test [0]
            else:
                if not os.path.exists('checkpoint'):
                    os.makedirs('checkpoint')
                save('checkpoint/bestmodel.pth', model, optimizer, scheduler, i, result_train[0] + result_valid[0] + result_test [0])
                tmp_score = result_train[0] + result_valid[0] + result_test [0]
        write_accuracy_to_csv(path1, result_train, result_valid, result_test)   
        
        

def eval_parallel(model, data_loader,optimizer, scheduler, is_training = True):
    mean_loss_E_V = []      
    acc_E_V = []  
    mean_loss_V_E = []      
    acc_V_E = []      
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    if is_training:
        model.train()
        data_iter = tqdm(data_loader, desc='Training', position=0, leave=True)     
        i = 1   
        for input, target in data_iter:
            optimizer.zero_grad()
            input, target = input.to(device), target.to(device)
            output = model(input, target, inp = 'E')
            loss = SparseCrossEntropy(target, output)           
            mean_loss_E_V.append(loss.item())
            acc_E_V.append(accuracy(target, output).item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            output = model(target, input, inp = 'V')          
            loss = SparseCrossEntropy(input, output)           
            mean_loss_V_E.append(loss.item())
            acc_V_E.append(accuracy(input, output).item())
            loss.backward()
            optimizer.step()                          
            scheduler.step()
            if i % 200 == 0:
                print(f"Vòng lặp thứ {i}, loss E_V {np.mean(np.array(mean_loss_E_V))}, acc E_V {np.mean(np.array(acc_E_V))}, loss V_E {np.mean(np.array(mean_loss_V_E))}, acc V_E {np.mean(np.array(acc_V_E))}")    
            i += 1       
            #if infer is not None:
                #infer = torch.cat([infer, output.to('cpu').argmax(dim = -1)], dim = 0)
            #else:
                #infer = output.to('cpu').argmax(dim = -1)
            #if candidate is not None:
                #candidate = torch.cat([candidate, target.to('cpu')], dim = 0)   
            #else:
                #candidate = target.to('cpu')                 
    else:
        model.eval()    
        with torch.no_grad():
            mean_loss_E_V = []      
            acc_E_V = []  
            mean_loss_V_E = []      
            acc_V_E = []    
            data_iter = tqdm(data_loader, desc='Not training', position=0, leave=True)        
            for input, target in data_iter:
                input, target = input.to(device), target.to(device)
                output = model(input, target, inp = 'E')      
                loss = SparseCrossEntropy(target, output)                     
                mean_loss_E_V.append(loss.item())
                acc_E_V.append(accuracy(target, output).item())
                output = model(target, input, inp = 'V')          
                loss = SparseCrossEntropy(input, output)           
                mean_loss_V_E.append(loss.item())
                acc_V_E.append(accuracy(input, output).item())
    return np.mean(np.array(mean_loss_E_V)), np.mean(np.array(acc_E_V)), np.mean(np.array(mean_loss_V_E)), np.mean(np.array(acc_V_E)) 

def train_parallel(model, optimizer, epoch, datatrain_loader,datavalid_loader = None, datatest_loader = None, path = "", path1 = ""):
    scheduler = LambdaLR(optimizer, lr_lambda= lambda step_num : transformer_lr(step_num=step_num, d_model=model.dmodel))
    tmp_score = 1e30
    if os.path.isfile(path):
        model, optimizer, scheduler, epoch_old, tmp_score = load(path, model, optimizer, scheduler)
    else:
        epoch_old = 0    
    for i in tqdm(range(epoch_old, epoch), desc='Epoch', position=0, leave=True):
        result_train =  eval_parallel(model, datatrain_loader, optimizer, scheduler, True)   
        if datavalid_loader is not None:
            result_valid = eval_parallel(model, datavalid_loader, optimizer, scheduler, False)
        else:
            result_valid = [0,0,0,0]
        if datatest_loader is not None:
            result_test = eval_parallel(model, datatest_loader, optimizer,scheduler, False)
        else:
            result_test = [0,0,0,0]            
        print(f"\n Train Accuracy E2V : {result_train[1]}, Train Accuracy V2E: {result_train[3]}, Validation Accuracy E2V: {result_valid[1]}, Validation Accuracy V2E: {result_valid[3]}, Test Accuracy E2V : {result_test[1]}, Test Accuracy V2E: {result_test[3]}, Train Loss E2V: {result_train[0]}, Train Loss V2E: {result_train[2]},Validation Loss E2V: {result_valid[0]}, Validation Loss V2E: {result_valid[2]},Test Loss E2V: {result_test[0]}, Test Loss V2E: {result_test[2]}")    
        if result_train[0] + result_valid[0] + result_test [0] < tmp_score:
            if path is not None:
                save(path, model, optimizer, scheduler, i, result_train[0] + result_valid[0] + result_test [0])
                tmp_score = result_train[0] + result_valid[0] + result_test [0]
            else:
                if not os.path.exists('checkpoint'):
                    os.makedirs('checkpoint')
                save('checkpoint/bestmodel.pth', model, optimizer, scheduler, i, result_train[0] + result_valid[0] + result_test [0])
                tmp_score = result_train[0] + result_valid[0] + result_test [0]
        write_accuracy_to_csv(path1, result_train, result_valid, result_test)                                     