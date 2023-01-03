from key import *
import glob
import openai
from gpt import GPT
from gpt import Example
import sys, os, argparse
import json
import torch
import shutil


# configure GPT
openai.api_key = key

# davinci = 0.06/1000 tokens
gpt = GPT(engine="davinci",
          temperature=0.5,
          output_prefix="Output: \n\n",
          max_tokens=1000)

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
        # print(weed_dataset_commands)
        # weed_dataset_commands = jsons_dict["water the plant species P"]
        weed_dataset_commands = json_dict["seed a plant P at location [[X,Y,Z]]"]
        # print(weed_dataset_commands)

        # embedding = openai.Embedding.create(weed_dataset_commands, engine='code-cushman-001')
        # print(embedding)

        for seq_of_function_calls in weed_dataset_commands[0:1]:
            # print(seq_of_function_calls["command"])
            # print(seq_of_function_calls["high_level_sequence_of_function_calls"])
            json_farmbot_bed = r'''[{"id": 754959, "created_at": "2022-02-01T23:15:32.859Z", "updated_at": "2022-04-28T06:18:00.850Z", "device_id": 13050, "name": "cilantro", "pointer_type": "Plant", "meta": {}, "x": 500, "y": 800, "z": -421, "openfarm_slug": "cilantro", "plant_stage": "sprouted", "planted_at": "2022-02-01T23:15:32.859Z", "radius": 150.0}, {"id": 754961, "created_at": "2022-02-01T23:15:32.859Z", "updated_at": "2022-04-28T06:20:47.723Z", "device_id": 13050, "name": "cilantro", "pointer_type": "Plant", "meta": {}, "x": 500, "y": 900, "z": -421, "openfarm_slug": "cilantro", "plant_stage": "sprouted", "planted_at": "2022-02-01T23:15:32.859Z", "radius": 150.0}, {"id": 754962, "created_at": "2022-02-01T23:15:32.859Z", "updated_at": "2022-04-28T06:21:02.385Z", "device_id": 13050, "name": "cilantro", "pointer_type": "Plant", "meta": {}, "x": 500, "y": 1100, "z": -421, "openfarm_slug": "cilantro", "plant_stage": "sprouted", "planted_at": "2022-02-01T23:15:32.859Z", "radius": 150.0}, {"id": 755254, "created_at": "2022-04-28T16:49:31.703Z", "updated_at": "2022-04-28T16:49:31.820Z", "device_id": 13050, "name": "Burpee Bush Tomato Big Boy", "pointer_type": "Plant", "meta": {}, "x": 1990, "y": 510, "z": 0, "openfarm_slug": "burpee-bush-tomato-big-boy", "plant_stage": "planned", "planted_at": "2022-04-28T16:49:31.703Z", "radius": 25.0}, {"id": 755253, "created_at": "2022-04-28T16:49:20.399Z", "updated_at": "2022-04-28T16:49:20.609Z", "device_id": 13050, "name": "Tiny Tim Tomato", "pointer_type": "Plant", "meta": {}, "x": 1270, "y": 530, "z": 0, "openfarm_slug": "tiny-tim-tomato", "plant_stage": "planned", "planted_at": "2022-04-28T16:49:20.399Z", "radius": 25.0}, {"id": 755255, "created_at": "2022-04-28T16:49:40.638Z", "updated_at": "2022-04-28T16:49:40.752Z", "device_id": 13050, "name": "UF Micro Tom Tomato", "pointer_type": "Plant", "meta": {}, "x": 2630, "y": 500, "z": 0, "openfarm_slug": "uf-micro-tom-tomato", "plant_stage": "planned", "planted_at": "2022-04-28T16:49:40.638Z", "radius": 25.0}, {"id": 755350, "created_at": "2022-02-01T23:15:32.859Z", "updated_at": "2022-06-02T22:07:30.467Z", "device_id": 13050, "name": "tim tomato", "pointer_type": "Plant", "meta": {"parent_family_name": "tomato"}, "x": 100, "y": 100, "z": -421, "openfarm_slug": "tiny-tim-tomato", "plant_stage": "planted", "planted_at": "2022-02-01T23:15:32.859Z", "radius": 225.0}, {"id": 755351, "created_at": "2022-02-01T23:15:32.859Z", "updated_at": "2022-06-02T22:07:43.537Z", "device_id": 13050, "name": "burpee bush tomato", "pointer_type": "Plant", "meta": {"parent_family_name": "tomato"}, "x": 200, "y": 200, "z": -421, "openfarm_slug": "burpee-bush-tomato-big-boy", "plant_stage": "planted", "planted_at": "2022-02-01T23:15:32.859Z", "radius": 225.0}, {"id": 755352, "created_at": "2022-02-01T23:15:32.859Z", "updated_at": "2022-06-02T22:08:11.563Z", "device_id": 13050, "name": "micro tom tomato", "pointer_type": "Plant", "meta": {"parent_family_name": "tomato"}, "x": 300, "y": 100, "z": -421, "openfarm_slug": "uf-micro-tom-tomato", "plant_stage": "planted", "planted_at": "2022-02-01T23:15:32.859Z", "radius": 225.0}, {"id": 560093, "created_at": "2021-07-01T18:46:06.783Z", "updated_at": "2022-04-28T05:14:51.146Z", "device_id": 13050, "name": "Sweet Pepper, California Wonder", "pointer_type": "Plant", "meta": {}, "x": 1900, "y": 1200, "z": -420, "openfarm_slug": "sweet-pepper-california-wonder", "plant_stage": "planned", "planted_at": "2021-07-01T18:46:06.783Z", "radius": 230.0}, {"id": 560094, "created_at": "2021-07-01T18:47:02.295Z", "updated_at": "2022-04-28T05:14:57.217Z", "device_id": 13050, "name": "Sweet Pepper, California Wonder", "pointer_type": "Plant", "meta": {}, "x": 1500, "y": 1200, "z": -420, "openfarm_slug": "sweet-pepper-california-wonder", "plant_stage": "planned", "planted_at": "2021-07-01T18:47:02.295Z", "radius": 230.0}, {"id": 633896, "created_at": "2021-10-13T19:43:01.101Z", "updated_at": "2021-10-13T19:43:01.453Z", "device_id": 13050, "name": "Red Carrot", "pointer_type": "Plant", "meta": {}, "x": 160, "y": 620, "z": 0, "openfarm_slug": "red-carrot", "plant_stage": "planned", "planted_at": "2021-10-13T19:43:01.101Z", "radius": 25.0}]'''

            all_points_farmbot_bed = json.loads(json_farmbot_bed)
            ultimate_string = ""
            #openai accepts an array of tokens, so we need to iterate through the json file and add each token to the array

            ultimate_list = []
            for plant in all_points_farmbot_bed:
                # ultimate_string = ultimate_string + str(plant)
                ultimate_list.append(str(plant))
            all_points_farmbot_bed = ultimate_list

            # print(all_points_farmbot_bed)
            # all_points_farmbot_bed = all_points_farmbot_bed.replace("{", " ")
            # all_points_farmbot_bed = all_points_farmbot_bed.replace("}", " ")
            # all_points_farmbot_bed = all_points_farmbot_bed.replace("False", " ")
            # print(all_points_farmbot_bed)
            #the openai api

            high_level_sequence_of_function_calls = seq_of_function_calls["high_level_sequence_of_function_calls"]

            natural_language_command = seq_of_function_calls["command"]
            natural_language_command_list = []
            for plant in all_points_farmbot_bed:
                # ultimate_string = ultimate_string + str(plant)
                natural_language_command_list.append(str(plant))
            natural_language_command = natural_language_command_list

            NL_embedding = openai.Embedding.create(
                input=natural_language_command,
                # engine="text-similarity-davinci-001"
                engine = "text-similarity-ada-001"
            )["data"][0]["embedding"]
            print("NL_embedding", len(NL_embedding))

            embedding_state = openai.Embedding.create(
                input=all_points_farmbot_bed,
                # engine="text-similarity-davinci-001"
                engine="text-similarity-ada-001"
            )["data"][0]["embedding"]
            print("embedding_state", len(embedding_state))

            print(embedding_state[0:5])

            embedding_answer = openai.Embedding.create(
                input=high_level_sequence_of_function_calls,
                engine="text-similarity-ada-001"
                # engine="text-similarity-davinci-001"
            )["data"][0]["embedding"]

            concatenated_embedding_input = NL_embedding + embedding_state
            print("concatenated_embedding_input", len(concatenated_embedding_input))

            # print(f"embedding_question: {embedding_question}")
            # print(len(embedding))

            # gpt.add_example(Example(str(natural_language_command), str(high_level_sequence_of_function_calls)))
            gpt.add_example(Example(str(concatenated_embedding_input), str(embedding_answer)))


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
    natural_language_command_prompt = "water 5 tomato plants"
    prompt_embedding = openai.Embedding.create(
        input=natural_language_command_prompt,
        # engine="text-similarity-davinci-001"
        # engine="text-similarity-ada-001"
        engine = "text-embedding-ada-002"

    )["data"][0]["embedding"]

    print("prompt_embedding", len(prompt_embedding))
    # prompt = str("water all of the Passion Fruit and wheat plants")

    output = gpt.get_top_reply(str(prompt_embedding))
    print(f"prompt_embedding: {prompt_embedding}")
    print(f"output: {output}")
    print("----------------------------------------")

