# Import libraries
import pandas as pd
import torch
import json

from torch.utils.data import Dataset, DataLoader

# create custom dataset class
class CustomTextDataset(Dataset):
    def __init__(self, nl_command, sequence_of_function_calls):  # (text,labels) => (nl,seq)
        self.nl_command = nl_command  # feed me all the nl_commands in a list
        self.sequence_of_function_calls = sequence_of_function_calls

    def __len__(self):
        return len(self.nl_command)  # length of set, not of an individual pair

    def __getitem__(self, idx):
        sequence_of_function_calls = self.sequence_of_function_calls[idx]  # label
        nl_command = self.nl_command[idx]  # data/text
        sample = {"nl_command": nl_command, "sequence_of_function_calls": sequence_of_function_calls}
        return sample

    # def preprocess(self, sequence, word2id, trg=True):
    #     """Converts words to ids."""
    #     tokens = nltk.tokenize.word_tokenize(sequence.lower())
    #     sequence = []
    #     sequence.append(word2id['<start>'])
    #     sequence.extend([word2id[token] for token in tokens if token in word2id])
    #     sequence.append(word2id['<end>'])
    #     sequence = torch.Tensor(sequence)
    #     return sequence

# define data and class labels
# with open("json_dataset.json", "r") as jsonDataset:
#     nl_commands = []
#     sequence_of_function_calls = []
#     dataset = json.load(jsonDataset)
#     for datapoint in dataset:
#         for nl_cmd in datapoint:
#         # listed_ultimate_json = list(datapoint.items()) # datapoint == ultimate_json == one state
#         # listed_final_function_call = list(listed_ultimate_json[-1])
#             nl_commands.append(nl_cmd)
#             sequence_of_function_calls.append(datapoint[nl_cmd])#[1][1]["history_of_function_calls"])
with open("list_of_input_output_sequences_only.json", "r") as jsonDataset:
    nl_commands = []
    sequence_of_function_calls = []
    dataset = json.load(jsonDataset)
    for datapoint in dataset:
        nl_commands.append(datapoint["input_sequence"])
        sequence_of_function_calls.append(datapoint["output_sequence"])

# text = ['Happy', 'Amazing', 'Sad', 'Unhappy', 'Glum']
# labels = ['Positive', 'Positive', 'Negative', 'Negative', 'Negative']

# create Pandas DataFrame
text_labels_df = pd.DataFrame({'nl_command': nl_commands, 'sequence_of_function_calls': sequence_of_function_calls})

# define data set object
TD = CustomTextDataset(text_labels_df['nl_command'], text_labels_df['sequence_of_function_calls'])

# Display image and label.
print('\nFirst iteration of data set: ', next(iter(TD)), '\n')

# Print how many items are in the data set
print('Length of data set: ', len(TD), '\n')

# Print entire data set
# print('Entire data set: ', list(DataLoader(TD)), '\n')


# collate_fn
def collate_batch(batch):
    nl_cmd_tensor = torch.tensor([[1.], [0.], [45.]])
    sequence_tensor = torch.tensor([[1.], [0.], [45.]])

    nl_cmd_list, sequences = [], []

    for (_text, _class) in batch:
        nl_cmd_list.append(nl_cmd_tensor)
        sequences.append(sequence_tensor)

    nl_cmds = torch.cat(nl_cmd_list)
    sequence_of_fcn_calls = torch.tensor(sequences)

    return nl_cmds, sequence_of_fcn_calls

# create DataLoader object of DataSet object
bat_size = 2
DL_DS = DataLoader(TD, batch_size=bat_size, shuffle=True)

# loop through each batch in the DataLoader object
for (idx, batch) in enumerate(DL_DS):
    pass
    # Print the 'text' data of the batch
    # print(idx, 'Text data: ', batch, '\n')

    # Print the 'class' data of batch
    # print(idx, 'Class data: ', batch, '\n')

# https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00
