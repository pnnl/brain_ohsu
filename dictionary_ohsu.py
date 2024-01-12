import re
import json
import os

with open('dict_names.json') as json_file:
    dict_model = json.load(json_file)

suffix = "test_1"
def lookup(s, lookups):
    for key, value in lookups.items():
        # if key matches regex
        re_match_object = re.search(r'{}'.format(key), s)
        if re_match_object:

            #sub in suffix name
            return value + re_match_object.groups()[0]
    return None


output_file = "/qfs/projects/brain_ohsu/TRAIL_MAP/dec723/data/testing/training__test_1_"

new_name = lookup(os.path.basename(output_file), dict_model)

print(dict_model[new_name])