# Aug 6 2018
# create training set for text classification

import csv
import sys
import math
import pickle
import json
import os
import argparse
import re
import numpy as np
from operator import itemgetter

def get_args():
    parser = argparse.ArgumentParser()

        ## Required parameters
    parser.add_argument("--output_dir",
                        default="data",
                        type=str,
                        help="Where to save the output data.")
    parser.add_argument("--input_data",
                        default="patent2",
                        type=str,
                        help="The input data dir.")
    parser.add_argument("--use_abstract", type=int, default=1)
    parser.add_argument("--use_claims",   type=int, default=1)
    parser.add_argument("--swap",   type=int, default=0)
    parser.add_argument("--invert",   type=int, default=0)
    parser.add_argument("--cat",   type=str, default="H04")
    parser.add_argument("--all_sub",   type=int, default=0)
    parser.add_argument('--text_length',
                        type=int,
                        default=-1,
                        help="How many characters will be in each datapoint.")
    return parser.parse_args()

def get_data_dir(args):
    if len(args.cat) != 3:
        exit("your category label is invalid")

    if args.all_sub == 1:
        filename = "uspto_all_subclass"
    else:
        filename = "uspto_{}".format(args.cat)

    if args.swap + args.invert == 2:
        exit("cant invert and swap the claims")

    if args.use_abstract == 1:
        filename += "_abstract"

    if args.use_claims == 1:
        filename += "_claims"

    if args.swap == 1:
        filename += "_swapped"

    if args.invert == 1:
        filename += "_inverted"

    return filename



def write_tsv(filename, data, args):

    list_data = [['patent_id','texta','label']]
    abstract_lens = []
    full_text_lens = []
    for k in data.keys():
        if args.invert == 1:
            full_text = " ".join([i[0] for i in data[k][3]]) + " " + data[k][2] + ". " +  data[k][1]
        elif args.swap == 1:
            full_text = " ".join([i[0] for i in data[k][3]]) + " " + data[k][1] + ". " +  data[k][2]
        else: #title + abstract + claims
            full_text = data[k][1] + ". " +  data[k][2] + " " + " ".join([i[0] for i in data[k][3]])
        list_data.append([k, full_text.replace("\t", " "), data[k][0]])

        abstract_lens.append(len(data[k][2]))
        full_text_lens.append(len(full_text))

    print("average len of abstracts: {}, full_text: {}".format(np.mean(np.array(abstract_lens)),np.mean(np.array(full_text_lens))))
    with open(filename, 'w', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerows(list_data)


def write_tsv_old(filename, data):
    with open(filename, 'w', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerows(data)

def is_same_cat(labels):
    if labels==[]:
        return None
    
    xs = [x[0] for x in labels]
    cat_label = xs[0]
    for z in xs[1:]:
        if z!=cat_label:
            return None
    return cat_label


def is_my_cat(labels, mycat):
    my_labels = [x for x in labels if x[:3]==mycat]
    if my_labels==[] or len(my_labels)>1:
        return None
    else:
        return my_labels[0]

#returns a dictionary 
#key: pid
#value: tuple {label, title, abstract, claims_list}
def parse_firstclaim(filename, mycat, all_sub=False):

    ret = {}
    count = 0
    all_labels = set()

    with open(filename, 'r') as f:
        for patent in json.load(f):
            labels = patent['cpc_ids'].split(',')

            cur_label = labels[0] if all_sub else is_my_cat(labels, mycat)

            if cur_label:
                if 'abstract' not in patent:
                    count +=1
                    continue
                title = patent['title']
                all_labels.add(cur_label)

                #remove weird stuff from the abstract
                abstract = patent['abstract'] 
                abstract =  re.sub('\(<b>.*?<\/b>\)', '', abstract)
                abstract =  re.sub('<b>.*?<\/b>', '', abstract)
                claims = [] #empty list to fill in later
                textb = patent['text'] #this is the first claim, and we dont use it anymore
                ret[patent['id']] = (cur_label, title, abstract, claims)

    print("file: {} missing {} abstracts".format(filename,count))
    print("all labels: {}".format(list(all_labels)))
    return ret , all_labels

def parse_fullclaim(filename, patent_dict, args):
    missing_dict = {}
    count = 0
    with open(filename, 'r') as f:
        for patent in json.load(f):
            if patent['id'] in patent_dict:
                #get all the claims for our cur patent
                try:
                    patent_dict[patent['id']][3].append((patent['text'],int(patent['sequence'])))
                except KeyError:
                    missing_dict[patent['id']] = 1
                    count +=1
    

    for k in patent_dict.keys():
        #sort the claims
        patent_dict[k][3].sort(key=itemgetter(1), reverse=(args.invert == 1))

    print("patents missing claims: {}. Total missing claims {}".format(len(missing_dict.keys()), count))
    return patent_dict


#debug_file_firstclaim = 'test_firstclaim.json'
#debug_file_allclaim = 'test_allclaim.json'

#train_dict = parse_firstclaim(debug_file_firstclaim)
#train_dict = parse_fullclaim(debug_file_allclaim, train_dict)

def main():
    args = get_args()


    output_dir_base = os.path.join(args.output_dir, get_data_dir(args))
    os.makedirs(output_dir_base, exist_ok=True)

    output_unlabeled_filename = os.path.join(output_dir_base,'corpus.txt')
    output_train_filename     = os.path.join(output_dir_base,'train.tsv') 
    output_test_filename      = os.path.join(output_dir_base,'test.tsv') 

    train_file_firstclaim = 'parsable_data/uspto_firstclaim_2006_2014.json'
    train_file_allclaims  = 'parsable_data/uspto_allclaims_2006_2014.json'
    test_file_firstclaim  = 'parsable_data/uspto_firstclaim_2015.json'
    test_file_allclaims   = 'parsable_data/uspto_allclaims_2015.json'


    print("parsing first claim test file")
    test_dict, test_keys = parse_firstclaim(test_file_firstclaim, args.cat, args.all_sub == 1)
    print("parsing full claim test file")
    test_dict = parse_fullclaim(test_file_allclaims, test_dict, args)


    print("saving test file")
    write_tsv(output_test_filename , test_dict, args)


    print("parsing first claim train file")
    train_dict, train_keys = parse_firstclaim(train_file_firstclaim, args.cat, args.all_sub == 1)
    print("parsing full claim train file")
    train_dict = parse_fullclaim(train_file_allclaims, train_dict,args )

    print("saving train file")
    write_tsv(output_train_filename, train_dict, args)

    if test_keys != train_keys:
        print("ERROR: test and train set have mismatching labels, respectively shown here:")
        print(test_keys)
        print(train_keys)
        print("keys combined:")
        print(list(test_keys.union(train_keys)))

    '''with open(output_unlabeled_filename, 'w') as file:
        for doc in train_data:
            file.write(doc[1].replace("\n", " ") + '\n')
            file.write(doc[2].replace("\n", " ") + '\n')         
            file.write('\n')
    '''



if __name__ == "__main__":
    main()
