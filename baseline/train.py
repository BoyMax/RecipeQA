import argparse
from tqdm import tqdm
from utils import recipeDataset, collate_batch_wrapper
from ImpatientReader import ImpatientReaderModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--doc_hidden_size", type=int, default=256)
    parser.add_argument("--question_hidden_size", type=int, default=256)
    parser.add_argument("--choice_hidden_size", type=int, default=256) #choice_hidden_size == attention output dimension
    parser.add_argument("--attention_hidden_size", type=int, default=256) # m_features
    parser.add_argument("--log_path", type=str, default="result/log_data.txt")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_model", type=str, default=None)
    args = parser.parse_args() 
    return args

def accuracy(preds, y):
    y = torch.LongTensor(y)
    preds = F.softmax(preds, dim=1)
    correct = 0 
    pred = preds.max(1, keepdim=True)[1]
    correct += pred.eq(y.view_as(pred)).sum().item()
    acc = correct/len(y)
    return acc

def train(args):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    #### Initialization
    # initialize model
    model = ImpatientReaderModel(args.doc_hidden_size, args.question_hidden_size, args.attention_hidden_size, args.choice_hidden_size, args.choice_hidden_size)
    model = model.to(device)
    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # initialize loss function
    criterion = nn.CrossEntropyLoss()

    ### loop epochs
    max_val_acc = 0
    for epoch in tqdm(range(args.num_epochs)):
        ### training
        #1. get training data
        train_dataset = recipeDataset(cleanFile='../data/hierarchy/train_cleaned.json', rawFile='../data/train.json', task='textual_cloze', structure='hierarchy')
        train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch_wrapper)
        #2. training all batches
        model.train()
        running_loss = 0
        running_acc = 0
        for batch_index, (text, image, question, choice, answer) in tqdm(enumerate(train_loader)):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(text, question, choice) #output is a list, length is 4, each element contains batch_size similarity scores
            outputs = torch.cat(outputs, 0).view(-1, len(answer)).permute(1, 0) # concatenate tensors in the list and transpose it as (batch_size, len_choice)
            loss = criterion(outputs, torch.LongTensor(answer).to(device)) #outputs:(batch_size, num_classes_similarity), answer:(batch)
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item() 
            running_acc += accuracy(outputs, answer)
        
        ### validating:
        #1. get validation data
        val_dataset = recipeDataset(cleanFile='./data/hierarchy/val_cleaned.json', rawFile='./data/val.json', task='textual_cloze', structure='hierarchy')
        val_loader = Data.DataLoader(val_dataset, batch_size=2, shuffle=True, collate_fn=collate_hierarchy_wrapper)
        #2. validation all batches
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch_index, (text, image, question, choice, answer) in enumerate(val_loader):
                # forward + compute loss and accuracy
                outputs = model(text, question, choice) #output is a list, length is 4, each element contains batch_size similarity scores
                outputs = torch.cat(outputs, 0).view(-1, args.batch_size).permute(1, 0) # concatenate tensors in the list and transpose it as (batch_size, len_choice)
                loss = criterion(outputs, torch.LongTensor(answer).to(device)) #outputs:(batch_size, num_classes_similarity), answer:(batch)
                # statistics
                val_loss += loss.item() 
                val_acc += accuracy(outputs, answer)

        # print every training epoch
        print('|Epoch %d | Training loss : %.3f | Training acc: %.2f | Validation loss: %.3f | Validation acc: %.2f' %
                epoch + 1, running_loss, running_acc, val_loss, val_acc)

        if val_acc > max_val_acc:
            max_val_acc = valid_acc
            save_model(model,epoch,val_acc,args.saved_path)

    print('Training Finished')


if __name__ == "__main__":
    train(get_args())
    