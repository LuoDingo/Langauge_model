#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:23:16 2020

@author: stevenalsheimer
"""

from fuzzywuzzy import fuzz
import csv
def String_check(string):
    Str1 = string
    Max_Ratio = 0
    Best_match = None
    with open('data_tolokers.csv','r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            line = row['sent']
            Str2 = line
            Ratio = fuzz.ratio(Str1.lower(),Str2.lower())
            if Ratio > Max_Ratio:
                Max_Ratio = Ratio
                Best_match = line
    if Max_Ratio < 60:
        return "No Match Here"
    else:
        return "input", string, "Did you mean:",Best_match, Max_Ratio

print(String_check("can help me"))

'''issues

matched I like apple to I like purple
matched fuck me to fuck u



'''