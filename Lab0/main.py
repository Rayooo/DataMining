from operator import itemgetter
import numpy as np
from collections import defaultdict

fileName = "affinity_dataset.txt"
data = np.loadtxt(fileName)

numApplePurchases = 0
for sample in data:
    if sample[3] == 1:
        numApplePurchases += 1

print(numApplePurchases)

features = ["bread","milk","cheese","apples","bananas"]
nFreatures = 5


validRules =  defaultdict(int)
invalidRules = defaultdict(int)
numOccurances = defaultdict(int)

for sample in data:
    for premise in range(nFreatures):
        if(sample[premise] == 0):
            continue
        numOccurances[premise] += 1

        for conclusion in range(nFreatures):
            if premise == conclusion:
                continue
            if sample[conclusion] == 1:
                validRules[(premise,conclusion)] += 1
            else:
                invalidRules[(premise,conclusion)] += 1


def print_rule(premise, conclusion, support, confidence, features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".format(premise_name, conclusion_name))
    print(" - Support: {0}".format(support[(premise,conclusion)]))
    print(" - Confidence: {0:.3f}".format(confidence[(premise,conclusion)]))

support = validRules
confidence = defaultdict(float)
for premise, conclusion in validRules.keys():
    rule = (premise, conclusion)
    confidence[rule] = validRules[rule] / float(numOccurances[premise])

for premise, conclusion in confidence:
   print_rule(premise, conclusion, support, confidence, features)

print("-----------------------------------------------------------")

sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)
for index in range(5):
    print("Rule #{0}".format(index + 1))
    premise, conclusion = sorted_support[index][0]
    print_rule(premise, conclusion, support, confidence, features)