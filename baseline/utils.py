from preprocessing import preprocess
import torch.utils.data as Data
import pdb
import torch

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
    return text, image, question, choice, answer


'''
## example of using customized dataloader.

train_dataset = recipeDataset(cleanFile='./data/hierarchy/train_cleaned.json', rawFile='./data/train.json', task='textual_cloze', structure='hierarchy')
loader = Data.DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=collate_hierarchy_wrapper)

for epoch in range(1):
    for recipe_index, (text, image, question, choice, answer) in enumerate(loader):
        if epoch == 0 and recipe_index == 0:
            print('Epoch:', epoch, '| recipe_index: ', recipe_index, '| context:',text , '| choice: ', choice)
            break
'''