import os
import sys
import re
import random
import json
from tqdm import tqdm
import time
import functools
import signal
from transformers import RobertaTokenizer
import matplotlib.pyplot as plt

f_train = open("train_err_cls.jsonl", 'w')
f_valid = open("valid_err_cls.jsonl", 'w')
f_test = open("test_err_cls.jsonl", 'w')
cont  = 0

directory_path = './Project_CodeNet_Python800/'
fuzz_dir_path = './fuzz/'

def read_err_file(path):
    input = open(path, 'r', encoding='utf-8').read()
    return input

def find_errorType(text):
    lines = text.splitlines()

# Extract the string before ":" in the fourth line
    if lines:
        last_line = lines[-1].strip()
        index_colon = last_line.find(':')

        if index_colon != -1:
            extracted_string = last_line[:index_colon].strip()
            return extracted_string
    else:
        return "No Error"

def find_errorLine(text):
    lines = text.splitlines()
    if len(lines) > 1:
        second_line_from_bottom = lines[-2].strip()
        return second_line_from_bottom
    else:
        return " "

# initialize for two plots
err_dict = {}

err_type_dict = {}
err_type_list = ['IndexError', 'ImportError', 'ValueError']
for i in err_type_list:
    err_type_dict[i] = 0

for dir in sorted(os.listdir(directory_path)):
    if dir != '.DS_Store':
        full_path = os.path.join(directory_path, dir)
        for f in os.listdir(full_path):
            item = os.path.join(f)
            # print(item)
            r = random.random()
            if r < 0.6:
                split = 'train'
            elif r < 0.8:
                split = 'valid'
            else:
                split = 'test'
            # js['index']=str(cont)
            err_dict[dir+'/'+item] = []

            fuzz_path = os.path.join(fuzz_dir_path, dir, item) + '/' + 'default/'
            if os.path.exists(fuzz_path):
                err_path = fuzz_path + 'stderr/'
                for file in sorted(os.listdir(err_path)):
                    err_file = os.path.join(err_path, file)

                    output = read_err_file(err_file)
                    error_str = find_errorType(output)
                    err_dict[dir+'/'+item].append(error_str)
                    
                    if error_str == 'IndexError':
                        err_type_dict['IndexError'] += 1
                    if error_str == 'ImportError':
                        err_type_dict['ImportError'] += 1
                    if error_str == 'ValueError':
                        err_type_dict['ValueError'] += 1

                    js={}

                    js['label']= error_str
                    js['error']= find_errorLine(output)
                    js['path'] = err_file
                    js['code']=open(os.path.join(full_path,item),encoding='latin-1').read()

                    # print(js['label'], js['error'], err_file)
                    if split == 'train':
                        f_train.write(json.dumps(js)+'\n')
                    elif split == 'valid':
                        f_valid.write(json.dumps(js)+'\n')
                    elif split == 'test':
                        f_test.write(json.dumps(js)+'\n')
                    else:
                        raise NotImplementedError
            # cont+=1

f_train.close()
f_valid.close()
f_test.close()

# this is the dict for the first plot
uniq_dict= {}
for i in range(6):
    uniq_dict[i] = 0

for key in err_dict:
    err_dict[key] = list(set(err_dict[key]))

    if 'No Error' not in err_dict[key]:
        uniq_dict[len(err_dict[key])] +=1
    else:
        if len(err_dict[key]) == 1:
            uniq_dict[0] += 1
        else: 
            uniq_dict[len(err_dict[key])-1] +=1
print(uniq_dict)

plt.bar(uniq_dict.keys(), uniq_dict.values(),  align='center', alpha=0.7)
plt.xlabel('number of unique error(s)')
plt.ylabel('number of solution(s)')
plt.title('Histogram of the number of unique error(s)')
plt.savefig('unique_errors.png')

plt.bar(err_type_dict.keys(), err_type_dict.values(), align='center', alpha=0.7)
plt.xlabel('Types of Error')
plt.ylabel('number of errors')
plt.title('Bar plot of the number of errors in each type')
plt.savefig('type_cnt_errors.png')

    


