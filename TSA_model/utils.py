from preprocessing import preprocess
import torch.utils.data as Data
import json
import torch
from random import choice
import pandas as pd
import numpy as np

class recipeDataset(Data.Dataset):
    def __init__(self, cleanFile,rawFile, task, structure):
        text, images, question, choices, answer = preprocess(cleanFile,rawFile, task, structure)  # other parameters use as default parameters
        self.text = text
        self.images = images
        self.question = question
        self.choices = choices
        self.answer = answer
    def __len__(self):
        return len(self.answer) # the number of recipe
    def __getitem__(self, index):
        text = self.text[index]
        images = self.images[index]
        question = self.question[index]
        choices = self.choices[index]
        answer = self.answer[index]
        return text, images, question, choices, answer

# padding before make batch(collate_hierarchy_wrapper) 
def padding_steps(content, type):  #type could be batch_first, or step_first(for hierarchy structure)
    # padding 
    # 1.find the max step
    max_step = len(content[0])
    for sample in content:
        if len(sample)>max_step:
            max_step = len(sample)
    
    # 2.1 batch_first : padding to get the same steps [recipes[steps[words]]]
    if type == "batch_first":
        for sample in content:
            if len(sample)<max_step:
                void_step = max_step - len(sample)
                for i in range(void_step):
                    sample.append(['0'])
        
    # 2.2 step_first padding to get the same steps [steps[recipes[words]]] ---for 'hierarchy'
    elif type == "step_first":
        transposed_recipes = []
        for step in range(max_step):
            step_recipes = []
            for sample in content:
                if len(sample)<=step:
                    step_recipes.append(['0'])
                else:
                    step_recipes.append(sample[step])
            transposed_recipes.append(step_recipes)
        content = transposed_recipes
    return content

# padding imgs to make all the recipes have the same number of images
def padding_imgs(content):  #type could be batch_first, or step_first(for hierarchy structure)
    # padding 
    # 1.find the max step
    max_pictures = len(content[0])
    for sample in content:
        if len(sample)>max_pictures:
            max_pictures = len(sample)
    
    # batch_first : padding to get the same steps [recipes[imgs]]
    for sample in content:
        if len(sample)<max_pictures:
            void_step = max_pictures - len(sample)
            for i in range(void_step):
                sample.append('0')
    return content

def replace_placeholder_with_choice(question, replaced_choice):
    placeholder = ['placeholder']
    return [replaced_choice if step == placeholder else step for step in question] # build a new list that replace placeholder with choice.

def batch_replace_answer(question_list, choice_list, answer_list): # question(batch, num_step, str) choice (batch, num_choice, str)  answer(batch, int)
    replaced_questions = []
    right_choices = []
    wrong_choices = []
    a = [1,2,3]
    b = [0,2,3] 
    c = [0,1,3]
    d = [0,1,2]
    sample_index = 0
    for i in range(len(answer_list)):
        # get choice corresponding to true answer
        right_answer_index = answer_list[i]
        # replace the placeholder with the right answer in the question
        right_choice = choice_list[i][right_answer_index]
        right_choices.append(replace_placeholder_with_choice(question_list[i], right_choice))
        # get only 1 wrong answer by sampling the wrong answer space
        if right_answer_index == 0:
            sample_index = choice(a)
        elif right_answer_index == 1:
            sample_index = choice(b)
        elif right_answer_index == 2:
            sample_index = choice(c)
        elif right_answer_index == 3:
            sample_index = choice(d)
        wrong_choice = choice_list[i][sample_index]
        wrong_choices.append(replace_placeholder_with_choice(question_list[i], wrong_choice))
    replaced_questions.append(right_choices)
    replaced_questions.append(wrong_choices)
    return replaced_questions #(2, batch_size, question_len, word_len)

def hierarchy_replace_answer(question_list, choice_list, answer_list): # question(num_step, batch, str) choice (batch, num_choice, str)  answer(batch, int)
    replaced_questions = []
    a = [1,2,3]
    b = [0,2,3] 
    c = [0,1,3]
    d = [0,1,2]
    sample_index = 0
    right_choices = []
    wrong_choices = []
    for step in question_list: 
        question_right_step = []
        question_wrong_step = []
        for batch in range(len(answer_list)):
            right_answer_index = answer_list[batch]
            if step[batch]== ['placeholder']:
                # add right choice in question
                question_right_step.append(choice_list[right_answer_index][batch])
                # add wrong choice in question
                if right_answer_index == 0:
                    sample_index = choice(a)
                elif right_answer_index == 1:
                    sample_index = choice(b)
                elif right_answer_index == 2:
                    sample_index = choice(c)
                elif right_answer_index == 3:
                    sample_index = choice(d)
                question_wrong_step.append(choice_list[sample_index][batch])
            else:
                question_right_step.append(step[batch])
                question_wrong_step.append(step[batch])
        right_choices.append(question_right_step)
        wrong_choices.append(question_wrong_step)
    replaced_questions.append(right_choices)
    replaced_questions.append(wrong_choices)
    return replaced_questions #(2, question_len, batch_size, word_len)

def collate_batch_wrapper(batch):
    transposed_data = list(zip(*batch))
    text = list(transposed_data[0])
    image = list(transposed_data[1])
    question = list(transposed_data[2])
    choice = list(transposed_data[3])
    answer = list(transposed_data[4])
    # padding steps for text and question
    text = padding_steps(text, "batch_first")
    question = padding_steps(question, "batch_first")
    return text, image, question, choice, answer

'''
train_dataset = recipeDataset('./data/entirety/train_cleaned.json')
loader = Data.DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_batch_wrapper, num_workers=2)
'''

def collate_hierarchy_wrapper(batch):
    transposed_data = list(zip(*batch))
    text = list(transposed_data[0])
    image = list(transposed_data[1])
    question = list(transposed_data[2])
    choice = list(transposed_data[3])
    answer = list(transposed_data[4])
    # padding steps for text and question
    text = padding_steps(text, "step_first")
    question = padding_steps(question, "step_first")
    choice = padding_steps(choice, "step_first")
    image = padding_imgs(image)
    return text, image, question, choice, answer

def collate_batch_hingeRank_wrapper(batch):
    transposed_data = list(zip(*batch))
    text = list(transposed_data[0])
    image = list(transposed_data[1])
    question = list(transposed_data[2])
    answer = list(transposed_data[4])
    # padding steps for text and question
    text = padding_steps(text, "batch_first")
    question = padding_steps(question, "batch_first")
    # get the right and wrong answer for hinge rank loss.
    choice_list = list(transposed_data[3])
    replaced_choice = batch_replace_answer(question, choice_list, answer) #question and choice shape must be (batch, step_len, word_len)
    return text, image, question, choice_list, answer, replaced_choice

def collate_hierarchy_hingeRank_wrapper(batch):
    transposed_data = list(zip(*batch))
    text = list(transposed_data[0])
    image = list(transposed_data[1])
    question = list(transposed_data[2])
    answer = list(transposed_data[4])
    # padding steps for text and question
    text = padding_steps(text, "step_first")
    question = padding_steps(question, "step_first")
    # get the right and wrong answer for hinge rank loss.
    choice_list = list(transposed_data[3])
    choice_list = padding_steps(choice_list, "step_first")
    replaced_choice = hierarchy_replace_answer(question, choice_list, answer) #question and choice shape must be (step_len, batch, word_len)
    return text, image, question, choice_list, answer, replaced_choice


def extract_image_feature(image, feature_path='../data/training_features_resnet50.json'):
    '''
    df = pd.read_json(feature_path, lines=True, chunksize=1e5)
    features = pd.DataFrame() # Initialize the dataframe
    '''
    f = open(feature_path, 'r', encoding='utf8').read()
    features = json.loads(f)
    image_features = []
    try:
        '''
        for df_chunk in df:
            features = pd.concat([features, df_chunk])
        '''
        for recipe_imgs in image:
            imgs_feature = []
            for img in recipe_imgs:
                if img == '0':
                    imgs_feature.append(np.zeros(1000).tolist())
                else:
                    '''
                    imgs_feature.append(features[img][0][0])
                    '''
                    imgs_feature.append(features[img][0])
            image_features.append(imgs_feature)
        if torch.cuda.is_available():
            image_features = torch.FloatTensor(image_features).cuda()
        else:
            image_features = torch.FloatTensor(image_features)
        return image_features
    except ValueError:
        print ('\nSome messages in the file cannot be parsed')

'''
## example of using customized dataloader.

train_dataset = recipeDataset(cleanFile='../data/hierarchy/train_cleaned.json', rawFile='../data/train.json', task='textual_cloze', structure='hierarchy')
loader = Data.DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=collate_hierarchy_wrapper)

for epoch in range(1):
    for recipe_index, (text, image, question, choice, answer) in enumerate(loader):
        if epoch == 0 and recipe_index == 0:
            print('\"question\": ',question)
            print('\"image\": ',image) 
            #print('\"replaced_choice\": ', replaced_choice)
            break
'''