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

def log_data(log_path,test_loss,test_accuracy):
    file = open(log_path,'a')
    if torch.cuda.is_available():
        data = str(test_loss) +' '+ str(f'{test_accuracy:.4f}') 
    else:
        data = str(test_loss) + ' '+ str(f'{test_accuracy:.4f}') 
    file.write(data)
    file.write('\n')
    file.close()

def test(args):
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
    
    ### test with existing model
    if args.load_model is not None:
        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['run_loss']
    
    ### testing:
    #1. get testing data
    test_dataset = recipeDataset(cleanFile='../data/hierarchy/test_cleaned.json', rawFile='../data/test.json', task='textual_cloze', structure='hierarchy')
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_hierarchy_wrapper)
    
    '''
    if torch.cuda.is_available():
        # use open file for gcp
        f = open('../data/validation_features_resnet50.json', 'r', encoding='utf8').read()
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

    #2. testing all batches
    print('Testing process')
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch_index, (text, image, question, choice, answer) in tqdm(enumerate(test_loader)):
            #image_feature = extract_image_feature(image, img_features)
            image_feature=''
            answer = torch.LongTensor(answer).to(device)
            # forward + compute loss and accuracy
            outputs = model(text, image_feature, question, choice) #output is a list, length is 4, each element contains batch_size similarity scores
            outputs = torch.cat(outputs, 0).view(-1, len(answer)).permute(1, 0) # concatenate tensors in the list and transpose it as (batch_size, len_choice)
            testing_loss = criterion(outputs, answer)
            # statistics
            test_loss += testing_loss.item() 
            test_acc += accuracy(outputs, answer)
    epoch_test_loss = test_loss/len(test_loader)
    epoch_test_acc = test_acc/len(test_loader)

    log_data(args.log_path, epoch_test_loss, epoch_test_acc)
    # print every testing epoch
    print('|Epoch %d | Testing loss : %.4f | Testing acc: %.4f'  %
            (epoch + 1, epoch_test_loss, epoch_test_acc))

    print('Testing Finished')


if __name__ == "__main__":
    test(get_args())
    