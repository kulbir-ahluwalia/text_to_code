from rhaldar_codet5 import CodeT5Summ
import json

# model = CodeT5Summ(pretrained_model="gpt2")
model = CodeT5Summ(pretrained_model="codeparrot/codeparrot")
# model = CodeT5Summ(pretrained_model="Salesforce/codet5-base-multi-sum")


with open("list_of_input_output_sequences_only.json", "r") as jsonDataset:
    nl_commands = []
    sequence_of_function_calls = []
    dataset = json.load(jsonDataset)
    for datapoint in dataset:
        nl_commands.append(datapoint["input_sequence"])
        sequence_of_function_calls.append(str(datapoint["output_sequence"]))

'''
codes is a list of strings representing code snippets
descriptions is a list of strings repreenting their descriptions
Together they constitute one minibatch
'''
# codes = ["print('hello world')", "a = b + c", "maximum = x if x > y else y"]
# descriptions = ["print the message hello world", "add two numbers", "store the maximum of two numbers"]
# training loop
batch_size = 1
# codes = sequence_of_function_calls[0]
# descriptions = nl_commands[0]
# model.train_minibatch(codes, descriptions)
# for batch_idx in range(batch_size, len(nl_commands), batch_size):  # full loop
for batch_idx in range(batch_size, 100, batch_size):  # smaller loop for puny laptop
    print("batch training index: ", batch_idx-batch_size, "-", batch_idx)
    codes = sequence_of_function_calls[batch_idx-batch_size:batch_idx]
    # print(len(codes), codes)
    descriptions = nl_commands[batch_idx-batch_size:batch_idx]
    # print(len(descriptions), descriptions)
    # codes and descriptions must be of the same length
    # assert len(codes) == len(descriptions)

    # Trains one minibatch (Input is two lists of strings)
    print(codes, descriptions)
    model.train_minibatch(codes, descriptions)

model.save(outpath="fine-tuned-model-weeding")

# Translates one code snippet (Input is one string)
# test_code = "['handler.http_request_api_points(args)', 'find_home(handler)', 'mount_tool(handler, \"weeder\")', 'weed_plants(handler, [[548.2,121.9,-160.9]])', 'dismount_tool(handler, \"weeder\")']"
# print(model.summarize(test_code))

# test_code_batch = ["print('Error')", "avg = np.mean(arr)", "assert length == 5"]
test_code_batch = [
                    "['handler.http_request_api_points(args)', 'get_coords_of_all_plants_of_species_x(handler.all_points_farmbot_bed, \"Romanesco\")', 'find_home(handler)', 'mount_tool(handler, \"seeder\")', 'seed_plants(handler, [[978.5,150.4,-32.2]])', 'dismount_tool(handler, \"seeder\")']",
                    "['handler.http_request_api_points(args)', 'go_to_absolute_position_x_y_z(handler, [[309.3,517.0,-191.7]])']",
                    "['handler.http_request_api_points(args)', 'find_home(handler)', 'mount_tool(handler, \"weeder\")', 'weed_plants(handler, [[294.9,811.3,-68.3]])', 'dismount_tool(handler, \"weeder\")']"
                   ]

# Translates a minibatch of code snippets (Input is one list of strings)
print(model.batch_summarize(test_code_batch))




