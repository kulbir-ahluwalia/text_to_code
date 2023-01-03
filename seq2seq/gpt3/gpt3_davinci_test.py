from key import *
import glob
import openai
from gpt import GPT
from gpt import Example
import sys, os, argparse
import json
import numpy as np
import shutil


# configure GPT
openai.api_key = key

# davinci = 0.06/1000 tokens
gpt = GPT(engine="davinci",
          temperature=0.5,
          output_prefix="Output: \n\n",
          max_tokens=1100)


# engine = "text-embedding-ada-002"

# # # curie = 0.006/1000 tokens
# gpt = GPT(engine="text-embedding-ada-002",
#           temperature=0.5,
#           output_prefix="Output: \n\n",
#           max_tokens=7192)


# # curie = 0.006/1000 tokens
# gpt = GPT(engine="curie",
#           temperature=0.5,
#           output_prefix="Output: \n\n",
#           max_tokens=100)

# # add some code examples
# for file in glob.glob("examples/*"):
#     title = file.replace("_", " ")
#     with open(f"{file}", "r") as f:
#         code = f.read()
#         print(f"Adding example: {title}")
#         print(f"code is: {code}")
#     gpt.add_example(Example(title, code))


def add_example_gpt3(data_collection_args):
    with open(data_collection_args.dataset_path_template, 'r+') as json_file:
        # return_dict = {}

        json_file = (json_file.read())
        json_dict = json.loads(json_file)

        # weed_dataset_commands = json_dict["remove the weed at location [[X,Y,Z]]"]
        # weed_dataset_commands = json_dict["water the plant species P"]
        weed_dataset_commands = json_dict["create a garden named X with the plants Y,Z"]
        # weed_dataset_commands = json_dict["seed a plant P at location [[X,Y,Z]]"]
        # print(weed_dataset_commands)

        sample_list = list(np.random.choice(weed_dataset_commands, size=3))
        for seq_of_function_calls in sample_list:
            print(seq_of_function_calls["command"])
            print(seq_of_function_calls["high_level_sequence_of_function_calls"])


            natural_language_command = seq_of_function_calls["command"]
            high_level_sequence_of_function_calls = seq_of_function_calls["high_level_sequence_of_function_calls"]

            gpt.add_example(Example(str(natural_language_command), str(high_level_sequence_of_function_calls)))





# # add some calculation examples
# gpt.add_example(Example("add 3+5", "8"))
# gpt.add_example(Example("add 8+5", "13"))
# gpt.add_example(Example("add 50+25", "75"))

# # Inferences
# prompt = "sort list in python"
# output = gpt.get_top_reply(prompt)
# print(prompt, ":", output)
# print("----------------------------------------")
#
# prompt = "Code weather api in python"
# output = gpt.get_top_reply(prompt)
# print(prompt, ":", output)
# print("----------------------------------------")
#
# prompt = "What is 876+89"
# output = gpt.get_top_reply(prompt)
# print(prompt, ":", output)

# print("----------------------------------------")



if __name__ == '__main__':
    data_collection_parser = argparse.ArgumentParser()

    # parser.add_argument('--data_path', type=str, default='../data/logs/', help='path for data jsons')
    # parser.add_argument('--gold_configs_dir', type=str, default='../data/gold-configurations/',
    #                     help='path for gold config xmls')
    #
    # parser.add_argument('--aug_data_dir', type=str, default='../data/augmented/', help='path for aug data')
    #
    # parser.add_argument('--seed', type=int, default=1234, help='random seed')

    data_collection_parser.add_argument('--farmbot_name', type=str, default='greenhouse_vision',
                                        help='Name of the farmbot to be used. This sets the credentials ')
    data_collection_parser.add_argument('--natural_language_command', type=str, default='',
                                        help='Natural language command input by the user to the farmbot')
    data_collection_parser.add_argument('--sequence_of_function_calls', nargs="+", default=[],
                                        help='A list of sequence of function calls')
    data_collection_parser.add_argument('--os_json_save_path', type=str, default='',
                                        help='path to save the json file')
    data_collection_parser.add_argument('--dataset_path_template', type=str, default='',
                                        help='path to the json dataset')

    data_collection_args = data_collection_parser.parse_args()

    # read_json("../NLP_commands/template_command_mapping.json")
    add_example_gpt3(data_collection_args)

    # prompt = str("remove the weed at location [[1000,2000,10545]]")
    # prompt = str("water the plant species Passion Fruit")
    # prompt = str("please seed siebel berries at location[[71888.0, 52550.8, -181.0]] thank you")
    #
    # output = gpt.get_top_reply(prompt)
    # print(prompt)
    # print(output)
    # print("----------------------------------------")

    # prompt = str("water all of the Passion Fruit and wheat plants")
    prompt = str("plant a row of 5 tomatoes in between rows of 10 basil plants")
    # prompt = str("water the top row of tomatoes every 3 days for 10 days starting today")
    # prompt = str("create a garden named summer garden with the plants starwberry, lentils")


    output = gpt.get_top_reply(prompt)
    print(prompt)
    print(output)
    print("----------------------------------------")

