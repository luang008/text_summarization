#
#  Copyright (c) 2015, Facebook, Inc.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree. An additional grant
#  of patent rights can be found in the PATENTS file in the same directory.
#
#  Author: Alexander M Rush <srush@seas.harvard.edu>
#          Sumit Chopra <spchopra@fb.com>
#          Jason Weston <jase@fb.com>

"""Construct the DUC test set. """

import sys
import argparse
import glob
import re
import nltk.data
from nltk.tokenize.treebank import TreebankWordTokenizer
import gensim
import os
import numpy as np
#@lint-avoid-python-3-compatibility-imports

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer = TreebankWordTokenizer()
def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=
                                     argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sum_docs', help="Article directory.", type=str)
    parser.add_argument('--year', help="DUC year to process.", type=str)
    parser.add_argument('--result_docs', help="Reference directory.", type=str)
    parser.add_argument('--ref_dir',
                        help="Directory to output the references.", type=str)
    parser.add_argument('--sys_dir',
                        help="Directory to output the references.", type=str)
    parser.add_argument('--article_file',
                        help="File to output the article sentences..", type=str)
    parser.add_argument('--vec_dir',
                        help="Directory to output the w2v vec files", type=str)
    args = parser.parse_args(arguments)

    #refs = [open("{0}/task1_ref{1}.txt".format(args.ref_dir, i), "w")
    #        for i in range(4)]
    
    #prefix = open(args.sys_dir + "/task1_prefix.txt", "w")
    if args.year == "2003":
        files = glob.glob("{0}/*/*".format(args.sum_docs))
    else:
        files = glob.glob("{0}/*/*".format(args.sum_docs))
    files.sort()
    empty_count = 0
    total_count = 0
    year_dir = args.vec_dir + args.year + "/"
    model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    " generate idf dictionaries"
    df_dict = {}
    for f in files:
        visited = set()
        # Make input.
        mode = 0
        text = ""
        for l in open(f):
            if l.strip() in ["</P>", "<P>"]:
                continue
            if mode == 1 and l.strip() != "<P>":
                text += l.strip() + " "
            if l.strip() == "<TEXT>":
                mode = 1
        text = " ".join([w for w in text.split() if w[0] != "&"])

        sents = sent_detector.tokenize(text)
        for first in sents:
            first = " ".join(tokenizer.tokenize(first.lower()))
            if ")" in first or ("_" in first and args.year == "2003"):
                first = re.split(" ((--)|-|_) ", first, 1)[-1]
            first = first.replace("(", "-lrb-") \
                         .replace(")", "-rrb-").replace("_", ",")
            for w in first.split(" "):
                visited.add(w)
        for ww in visited:
            df_dict[ww] = df_dict.get(ww,0) + 1
    " generate vec"
    for f in files:
        #docset = f.split("/")[-2][:-1].upper()
        #name = f.split("/")[-1].upper()
        output_dir = year_dir + f.split("/")[-2]
        print output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_vec_path = output_dir + '/' + f.split("/")[-1] + ".vec"
        output_tfidf_path = output_dir + '/' + f.split("/")[-1] + ".tfidf"
        print output_vec_path
        fvec = open(output_vec_path, "w")
        ftfidf = open(output_tfidf_path,"w")

        # Make input.
        mode = 0
        text = ""
        for l in open(f):
            if l.strip() in ["</P>", "<P>"]:
                continue
            if mode == 1 and l.strip() != "<P>":
                text += l.strip() + " "
            if l.strip() == "<TEXT>":
                mode = 1
        text = " ".join([w for w in text.split() if w[0] != "&"])

        sents = sent_detector.tokenize(text)

        for first in sents:
            count = 0
            weight = 0
            #first = sents[0]
            '''
            # If the sentence is too short, add the second as well.
            if len(sents[0]) < 130 and len(sents) > 1:
                first = first.strip()[:-1] + " , " + sents[1]
            '''
            tf_dict = {}
            weight_dict = {}
            vec = np.zeros(300)
            tfidf_vec = np.zeros(300)
            
            first = " ".join(tokenizer.tokenize(first.lower()))
            if ")" in first or ("_" in first and args.year == "2003"):
                first = re.split(" ((--)|-|_) ", first, 1)[-1]
            first = first.replace("(", "-lrb-") \
                         .replace(")", "-rrb-").replace("_", ",")
            for w in first.split(" "):
                tf_dict[w] = tf_dict.get(w,0) + 1
            for w in tf_dict:
                weight_dict[w] = tf_dict[w] * 1.0 * np.log(1.0 * len(files)/df_dict[w] + 1.0)
            for w in first.split(" "):
                
                try:
                    v = model[w]
                except:
                    continue
                '''
                if w in model:
                    v = model[w]
                else:
                    continue
                '''
                vec += v
                tfidf_vec += weight_dict[w] * v
                count += 1
                weight += weight_dict[w]

            if count != 0:
                vec /= count
                tfidf_vec /= weight
            print >>fvec, ' '.join([str(x) for x in vec])
            print >>ftfidf, ' '.join([str(x) for x in tfidf_vec])
            #print >>article, first
            #print >>prefix, first[:75]
        fvec.close()
        ftfidf.close()
if __name__ == '__main__':
    main(sys.argv[1:])
