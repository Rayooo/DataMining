import twitter
import os
import json
import ssl

from sklearn.model_selection import cross_val_score

ssl._create_default_https_context = ssl._create_unverified_context

consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""
authorization = twitter.OAuth(access_token, access_token_secret, consumer_key, consumer_secret)

output_filename = os.path.join(os.path.expanduser("~/Documents/DataMining/DataMining/Lab2"), "python_tweets.json")

# t = twitter.Twitter(auth=authorization)
# with open(output_filename, 'a') as output_file:
#     search_results = t.search.tweets(q="python", count=100)['statuses']
#     for tweet in search_results:
#         if 'text' in tweet:
#             output_file.write(json.dumps(tweet))
#             output_file.write("\n\n")


input_filename = os.path.join(os.path.expanduser("~/Documents/DataMining/DataMining/Lab2"), "python_tweets.json")
labels_filename = os.path.join(os.path.expanduser("~/Documents/DataMining/DataMining/Lab2"), "python_classes.json")

tweets = []  # 创建列表，用于存储从文件中读进来的每条消息。
with open(input_filename) as inf:
    for line in inf:
        if len(line.strip()) == 0:  # 检测当前行（去除任意空白字符后的）长度是否为0
            continue
        tweets.append(json.loads(line))  # json.loads 将JSON字符串转换为Python对象

tweet_sample = tweets
labels = []
if os.path.exists(labels_filename):
    with open(labels_filename) as inf:
        labels = json.load(inf)


def get_tweet():
    return tweet_sample[len(labels)]['text']


tweet_sample = tweets
labels = []
if os.path.exists(labels_filename):
    with open(labels_filename) as inf:
        labels = json.load(inf)


def get_tweet():
    return tweet_sample[len(labels)]['text']


#
# %%html
# <div name='tweetbox'>
#     Instructions: Click in test box. Enter a 1 if the tweet is relevant, enter 0 otherwise. <br>
#     Tweet: <div id="tweet_text" value='text'></div> <br>
#     <input type="text" id="capture"> <br>
# </div>
# <script>
# function
# set_label(label)
# {
#     var
# kernel = IPython.notebook.kernel;
# kernel.execute('labels.append(' + label + ')');
# load_next_tweet();
# }
#
# function
# load_next_tweet()
# {
#     console.log('1');
# var
# code_input = 'get_tweet()';
# console.log('2');
# var
# kernel = IPython.notebook.kernel;
# console.log("3");
# var
# callbacks = {'iopub': {'output': handle_output}};
# console.log("4");
# kernel.execute(code_input, callbacks, {silent: false});
# console.log("5");
# }
#
# function
# handle_output(out)
# {
#     console.log(out);
# var
# res = out.content.data['text/plain'];
# $('div#tweet_text').html(res);
# }
# $("input#capture").keypress(function(e)
# {
#     console.log(e);
# if (e.which == 48)
# {
# // 0 pressed
# set_label(0);
# $("input#capture").val("");
# } else if (e.which == 49) {
# // 1 pressed
# set_label(1);
# $("input#capture").val("");
# }
# })
# load_next_tweet();
# </script>


with open(labels_filename, 'w') as outf:
    json.dump(labels, outf)

import os

replicable_dataset = os.path.join(os.path.expanduser("~/Documents/DataMining/DataMining/Lab2"),
                                  "replicable_dataset.json")
import json

tweets = []
with open(input_filename) as inf:
    for line in inf:
        if len(line.strip()) == 0:
            continue
        tweets.append(json.loads(line))
if os.path.exists(labels_filename):
    with open(labels_filename) as inf:
        labels = json.load(inf)
dataset = [(tweet['id'], label) for tweet, label in zip(tweets, labels)]
with open(replicable_dataset, 'w') as outf:
    json.dump(dataset, outf)

from sklearn.base import TransformerMixin


class NLTKBOW(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [{word: True for word in word_tokenize(document)} for document in X]


# 加载消息。我们只对消息内容感兴趣，因此只提取和存储它们的text值。代码如下：

tweets = []
with open(input_filename) as inf:
    for line in inf:
        if len(line.strip()) == 0:
            continue
        tweets.append(json.loads(line)['text'])
# 加载消息的类别。

with open(labels_filename) as inf:
    labels = json.load(inf)

# 创建流水线，把所有部件组合起来。流水线包含以下三个部分。
# 我们创建的NLTKBOW转换器  DictVectorizer转换器  BernoulliNB分类器

from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline

pipeline = Pipeline([('bag-of-words', NLTKBOW()), ('vectorizer', DictVectorizer()), ('naive-bayes', BernoulliNB())])

