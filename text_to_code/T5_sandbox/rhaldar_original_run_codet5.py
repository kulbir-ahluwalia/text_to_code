from rhaldar_original_codet5 import CodeT5Summ


model = CodeT5Summ()

'''
codes is a list of strings representing code snippets
descriptions is a list of strings repreenting their descriptions
Together they constitute one minibatch
'''
codes = ["print('hello world')", "a = b + c", "maximum = x if x > y else y"]
descriptions = ["print the message hello world", "add two numbers", "store the maximum of two numbers"]

# codes and descriptions must be of the same length
assert len(codes) == len(descriptions)

# Trains one minibatch (Input is two lists of strings)
model.train_minibatch(codes, descriptions)

test_code = "print('Error')"

# Translates one code snippet (Input is one string)
print(model.summarize(test_code))

test_code_batch = ["print('Error')", "avg = np.mean(arr)", "assert length == 5"]

# Translates a minibatch of code snippets (Input is one list of strings)
print(model.batch_summarize(test_code_batch))