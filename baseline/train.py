import argparse
from tqdm import tqdm
from utils import recipeDataset, collate_batch_hingeRank_wrapper
from ImpatientReader import ImpatientReaderModel, HingeRankLoss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--doc_hidden_size", type=int, default=256)
    parser.add_argument("--question_hidden_size", type=int, default=100) # for hinge rank loss: question_hidden_size = choice_hidden_size
    parser.add_argument("--choice_hidden_size", type=int, default=100) #choice_hidden_size
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

def save_model(epoch, model, optimizer,run_loss, val_loss, accuracy, saved_path):
    torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'run_loss': run_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            },
               '%s/IR_hingerank_epoch_%d_acc_%f.pth' % (saved_path, epoch,accuracy))
    print('Save model with accuracy:',accuracy)


def train(args):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    #### Initialization
    # initialize model
    model = ImpatientReaderModel(args.doc_hidden_size, args.question_hidden_size, args.attention_hidden_size, args.choice_hidden_size, args.choice_hidden_size)
    model = model.to(device)
    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # initialize loss function
    criterion = HingeRankLoss(margin=1.5)

    
    ### train with existing model
    if args.load_model is not None:
        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['run_loss']
    
    ### loop epochs
    max_val_acc = 0
    for epoch in tqdm(range(args.num_epochs)):
        ### training
        #1. get training data
        train_dataset = recipeDataset(cleanFile='../data/hierarchy/train_cleaned.json', rawFile='../data/train.json', task='textual_cloze', structure='hierarchy')
        train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch_hingeRank_wrapper)
        #2. training all batches
        model.train()
        running_loss = 0
        running_acc = 0
        for batch_index, (text, image, question, choice, answer, replaced_choice) in tqdm(enumerate(train_loader)):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs, g, r, w = model(text, question, choice, replaced_choice) #output is a list, length is 4, each element contains batch_size similarity scores
            outputs = torch.cat(outputs, 0).view(-1, len(answer)).permute(1, 0) # concatenate tensors in the list and transpose it as (batch_size, len_choice)
            loss = criterion(g, r, w) #outputs:(batch_size, num_classes_similarity), answer:(batch)
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item() 
            running_acc += accuracy(outputs, answer)
        epoch_train_loss = running_loss/len(train_loader)
        epoch_train_acc = running_acc/len(train_loader)

        ### validating:
        #1. get validation data
        val_dataset = recipeDataset(cleanFile='../data/hierarchy/val_cleaned.json', rawFile='../data/val.json', task='textual_cloze', structure='hierarchy')
        val_loader = Data.DataLoader(val_dataset, batch_size=10, shuffle=True, collate_fn=collate_batch_hingeRank_wrapper)
        #2. validation all batches
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch_index, (text, image, question, choice, answer, replaced_choice) in enumerate(val_loader):
                # forward + compute loss and accuracy
                outputs, g, r, w = model(text, question, choice, replaced_choice) #output is a list, length is 4, each element contains batch_size similarity scores
                outputs = torch.cat(outputs, 0).view(-1, len(answer)).permute(1, 0) # concatenate tensors in the list and transpose it as (batch_size, len_choice)
                validation_loss = criterion(g, r, w) #outputs:(batch_size, num_classes_similarity), answer:(batch)
                # statistics
                val_loss += validation_loss.item() 
                val_acc += accuracy(outputs, answer)
        epoch_val_loss = val_loss/len(val_loader)
        epoch_val_acc = val_acc/len(val_loader)

        # print every training epoch
        print('|Epoch %d | Training loss : %.3f | Training acc: %.2f | Validation loss: %.3f | Validation acc: %.2f' %
                (epoch + 1, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc))

        if epoch_val_loss > max_val_acc:
            max_val_acc = epoch_val_loss
            save_model(epoch, model, optimizer, loss, validation_loss, epoch_val_acc, args.saved_path)

    print('Training Finished')


if __name__ == "__main__":
    train(get_args())
    