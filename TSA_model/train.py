import argparse
from tqdm import tqdm
from utils import *
from TSAModel import TSAModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--doc_hidden_size", type=int, default=256)
    parser.add_argument("--img_hidden_size", type=int, default=256)
    parser.add_argument("--question_hidden_size", type=int, default=256) # for hinge rank loss: question_hidden_size = choice_hidden_size
    parser.add_argument("--choice_hidden_size", type=int, default=256) #choice_hidden_size
    parser.add_argument("--attention_hidden_size", type=int, default=256) # m_features
    parser.add_argument("--similarity_type", type=str, default="infersent") 
    parser.add_argument("--embed_hidden_size", type=int, default=256)
    parser.add_argument("--log_path", type=str, default="result.txt")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_model", type=str, default=None)
    args = parser.parse_args() 
    return args

def accuracy(preds, y):
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
               '%s/TSA_epoch_%d_acc_%f.pth' % (saved_path, epoch,accuracy))
    print('Save model with accuracy:',accuracy)

def log_data(log_path,train_loss,train_accuracy,val_loss,val_accuracy):
    file = open(log_path,'a')
    if torch.cuda.is_available():
        data = str(train_loss) +' '+ str(f'{train_accuracy:.2f}') \
            +' '+ str(val_loss)+ ' ' + str(f'{val_accuracy:.2f}')
    else:
        data = str(train_loss) + ' '+ str(f'{train_accuracy:.2f}') \
                +' '+str(val_loss)+' '+str(f'{val_accuracy:.2f}')
    file.write(data)
    file.write('\n')
    file.close()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #### Initialization
    # initialize model
    model = TSAModel(args.doc_hidden_size, args.img_hidden_size, args.question_hidden_size, args.attention_hidden_size, 
                                args.choice_hidden_size, args.choice_hidden_size, args.similarity_type,args.embed_hidden_size)
    model = model.to(device)
    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # initialize loss function
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
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
        train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_hierarchy_wrapper)

        '''
        if torch.cuda.is_available():
            # use open file for gcp
            f = open('../data/training_features_resnet50.json', 'r', encoding='utf8').read()
            img_features = json.loads(f)
        else:
            # use pandas.read_json for local machine
            df = pd.read_json('../data/training_features_resnet50.json', lines=True, chunksize=1e5)
            img_features = pd.DataFrame() # Initialize the dataframe
            try:
                for df_chunk in df:
                    img_features = pd.concat([img_features, df_chunk])
            except ValueError:
                print ('\nSome messages in the file cannot be parsed')
        '''

        #2. training all batches
        model.train()
        running_loss = 0
        running_acc = 0
        for batch_index, (text, image, question, choice, answer) in tqdm(enumerate(train_loader)):
            # extract image feature for batch image names
            # image (batch, image_len)
            # image_feature = extract_image_feature(image, img_features)
            image_feature=''

            answer = torch.LongTensor(answer).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(text, image_feature, question, choice) #output is a list, length is 4, each element contains batch_size similarity scores
            outputs = torch.cat(outputs, 0).view(-1, len(answer)).permute(1, 0) # concatenate tensors in the list and transpose it as (batch_size, len_choice)
            loss = criterion(outputs, answer)
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
        val_loader = Data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_hierarchy_wrapper)
        
        '''
        if torch.cuda.is_available():
            # use open file for gcp
            f = open('../data/training_features_resnet50.json', 'r', encoding='utf8').read()
            img_features = json.loads(f)
        else:
            # use pandas.read_json for local machine
            df = pd.read_json('../data/validation_features_resnet50.json', lines=True, chunksize=1e5)
            img_features = pd.DataFrame() # Initialize the dataframe
            try:
                for df_chunk in df:
                    img_features = pd.concat([img_features, df_chunk])
            except ValueError:
                print ('\nSome messages in the file cannot be parsed')
        '''

        #2. validation all batches
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch_index, (text, image, question, choice, answer) in enumerate(val_loader):
                # image_feature = extract_image_feature(image, img_features)
                image_feature=''

                answer = torch.LongTensor(answer).to(device)
                # forward + compute loss and accuracy
                outputs = model(text, image_feature, question, choice) #output is a list, length is 4, each element contains batch_size similarity scores
                outputs = torch.cat(outputs, 0).view(-1, len(answer)).permute(1, 0) # concatenate tensors in the list and transpose it as (batch_size, len_choice)
                validation_loss = criterion(outputs, answer)
                # statistics
                val_loss += validation_loss.item() 
                val_acc += accuracy(outputs, answer)
        epoch_val_loss = val_loss/len(val_loader)
        epoch_val_acc = val_acc/len(val_loader)

        log_data(args.log_path, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc)
        # print every training epoch
        print('|Epoch %d | Training loss : %.3f | Training acc: %.2f | Validation loss: %.3f | Validation acc: %.2f' %
                (epoch + 1, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc))

        if epoch_val_acc > max_val_acc:
            max_val_acc = epoch_val_acc
            save_model(epoch, model, optimizer, loss, validation_loss, epoch_val_acc, args.saved_path)

    print('Training Finished')


if __name__ == "__main__":
    train(get_args())
    