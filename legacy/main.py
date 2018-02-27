# https://en.wikipedia.org/wiki/List_of_newspapers_in_the_United_States
# http://michelleful.github.io/code-blog/2015/09/10/parsing-chinese-with-stanford/
# RAKE:
# https://www.airpair.com/nlp/keyword-extraction-tutorial
# https://github.com/zelandiya/RAKE-tutorial

# The following algorithms are implemented. Using #3 by default:
# 1. News clustering with MDS: compute the distance matrix (distance is the cosine value of two news vectors) and use MDS to reduce to 2D/3D for visualization.
# 2. News clustering with subgraph: only use a portion of the distances (e.g. the shortest 20%), divide news into subgraphs, each subgraph is a cluster.
# 3. Keyword graph: extract noun phrases from news, use page rank (an edge exists between any pair of noun phrases found in one news) to pick the top phrases as keywords.
# http://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/
# http://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
from collections import defaultdict, OrderedDict
from mpl_toolkits.mplot3d import Axes3D
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from rake import rake
from sklearn.manifold import MDS
from urllib.request import urlopen
import code
import feedparser
import html
import math
import matplotlib.pyplot as plt
import networkx
import nltk
import numpy as np
import os
import pandas as pd
import re
import sys

env = {}

# For NP extraction.
sentencePattern = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
grammar = r'''
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        
    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
'''
chunker = nltk.RegexpParser(grammar)
stopwords = stopwords.words('english')

# For RAKE.
raker = rake.Rake('rake/SmartStoplist.txt', 5, 3)

class NewsSource(object):
    def __init__(self, description, link, filecache=None, titleRe=None, summaryRe=None):
        self.description = description
        self.link = link
        self.filecache = filecache
        self.titleRe = titleRe
        self.summaryRe = summaryRe

class Translater(object):
    ''' Tool to translate special UTF8 char to ASCII. '''
    # Special UTF8 char to ASCII char mapping.
    chars = {
        '\xc2\x82' : ',',        # High code comma
        '\xc2\x84' : ',,',       # High code double comma
        '\xc2\x85' : '...',      # Tripple dot
        '\xc2\x88' : '^',        # High carat
        '\xc2\x91' : '\x27',     # Forward single quote
        '\xc2\x92' : '\x27',     # Reverse single quote
        '\xc2\x93' : '\x22',     # Forward double quote
        '\xc2\x94' : '\x22',     # Reverse double quote
        '\xc2\x95' : ' ',
        '\xc2\x96' : '-',        # High hyphen
        '\xc2\x97' : '--',       # Double hyphen
        '\xc2\x99' : ' ',
        '\xc2\xa0' : ' ',
        '\xc2\xa6' : '|',        # Split vertical bar
        '\xc2\xab' : '<<',       # Double less than
        '\xc2\xbb' : '>>',       # Double greater than
        '\xc2\xbc' : '1/4',      # one quarter
        '\xc2\xbd' : '1/2',      # one half
        '\xc2\xbe' : '3/4',      # three quarters
        '\xca\xbf' : '\x27',     # c-single quote
        '\xcc\xa8' : '',         # modifier - under curve
        '\xcc\xb1' : '',         # modifier - under line
        '\u2018'   : '\'',
        '\u2019'   : '\'',
        '\u201c'   : '"',
        '\u201d'   : '"',
        '\xa0'     : ' ',
    }

    utf2asciiPattern = '(' + '|'.join(chars.keys()) + ')'

    @classmethod
    def Translate(cls, content):
        return re.sub(Translater.utf2asciiPattern, lambda x: Translater.chars[x.group(0)], content)

class NewsItem(object):
    def __init__(self, source, entry):
        self.title = re.sub('<[^<]+?>', '', Translater.Translate(entry.title))
        self.summary = re.sub('<[^<]+?>', '', Translater.Translate(entry.summary))
        if source.titleRe is not None:
            self.title = source.titleRe(self.title)
        if source.summaryRe is not None:
            self.summary = source.summaryRe(self.summary)
        self.kwt = [x[0] for x in raker.run(self.title)]
        self.kws = [x[0] for x in raker.run(self.summary)]
        self.nps = set(self.kwt + self.kws)
        # Use NP extraction.
        # self.nps = self.ExtractNP(self.title) | self.ExtractNP(self.summary)
        self.link = entry.link
        self.source = source
        # DEBUG
        # open('inspect.txt', 'a', encoding='utf-8').write('{0}\n{1}\n{2}\n==========\n{3}\n++++++++++\n{4}\n{5}\n==========\n\n'.format(self.source.description, self.title, self.summary, list(self.nps), self.kwt, self.kws))
        # DEBUG
        '''
        lmtzr = WordNetLemmatizer()
        tokenizer = RegexpTokenizer(r'\w+')
        words = [lmtzr.lemmatize(x.lower()) for x in tokenizer.tokenize(self.summary) + tokenizer.tokenize(self.title) if x.lower() not in stopwords.words('english')]
        self.words = set(words)
        self.wordCount = len(words)
        self.hist = defaultdict(int)
        for word in words:
            self.hist[word] += 1
        '''

    def ExtractNP(self, text):
        toks = nltk.regexp_tokenize(text, sentencePattern)
        postoks = nltk.tag.pos_tag(toks)
        tree = chunker.parse(postoks)
        nps = set([])
        for leaf in self.__Leaves(tree):
            try:
                terms = [self.__Normalise(w) for w,t in leaf if self.__IsAcceptable(w)]
            except Exception as e:
                print(e)
                code.interact(local=locals())
            if len(terms) > 0:
                nps.add(' '.join(terms))
        return nps

    # Finds NP leaf nodes of a chunk tree.
    def __Leaves(self, tree):
        for subtree in tree.subtrees(filter = lambda t: t.label() == 'NP'):
            yield subtree.leaves()

    # Normalise words to lowercase, stem or lemmatize.
    def __Normalise(self, word):
        word = word.lower()
        # word = stemmer.stem_word(word) #if we consider stemmer then results comes with stemmed word, but in this case word will not match with comment
        word = lemmatizer.lemmatize(word)
        return word

    # Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase.
    def __IsAcceptable(self, word):
        return bool(2 <= len(word) <= 40 and word.lower() not in stopwords)

def TextRank(n, edges):
    # Build graph.
    graph = networkx.Graph()
    graph.add_nodes_from(range(n)) # Add nodes.
    graph.add_edges_from(edges) # Add deges.

    # Rank nodes.
    ranks = networkx.pagerank(graph) # Page rank.
    sortedNodes = [x[0] for x in sorted(ranks.items(), key=lambda x: x[1], reverse=True)]
    return sortedNodes

# Compute distance of two news items. Distance = cos(itemI, itemJ) = vec(ItemI) * vec(ItemJ) / (||vec(ItemI)|| * ||vec(ItemJ)||)
def Distance(itemI, itemJ):
    commonWords = itemI.words & itemJ.words
    return 1 - float(sum(itemI.hist[x] * itemJ.hist[x] for x in commonWords)) / (math.sqrt(sum(itemI.hist[x] ** 2 for x in itemI.words)) * math.sqrt(sum(itemJ.hist[x] ** 2 for x in itemJ.words)))

def FindSubgraphs(n, edges):
    remainingVertices = set([x for x in range(n)])
    subgraphs = []
    while len(remainingVertices) > 0:
        subgraph = set([])
        boundary = set([next(iter(remainingVertices))])
        while len(boundary) > 0:
            subgraph |= boundary
            remainingVertices -= boundary
            boundary = remainingVertices & (set(edges[edges['j'].isin(boundary)]['i']) | set(edges[edges['i'].isin(boundary)]['j']))
        subgraphs.append(subgraph)
    return subgraphs

# Load all configurations.
def LoadConfig():
    global env
    env = {
        'debug': True,
        'forceCacheRefresh': False,
        'isCacheEnabled': True,
        'newsGroups': {
        }
    }

    env['newsGroups']['politics'] = [
        NewsSource('BBC', 'http://feeds.bbci.co.uk/news/politics/rss.xml', 'BBC_politics.html'),
        NewsSource('NPR', 'https://www.npr.org/rss/rss.php?id=1014', 'NPR_politics.html'),
        NewsSource('New York Times', 'http://www.nytimes.com/services/xml/rss/nyt/Politics.xml', 'NYT_politics.html'),
        NewsSource('Fox News', 'http://feeds.foxnews.com/foxnews/politics', 'FN_politics.html'),
        NewsSource('The Guardian', 'https://www.theguardian.com/us-news/us-politics/rss', 'TG_politics.html', lambda x: re.sub('– video', '', x), lambda x: re.sub('Continue reading\.\.\.|Related: .*$', '', x)),
        #NewsSource('New York Daily News', 'http://www.nydailynews.com/cmlink/NYDN.News.Politics.rss',  'NYDN_politics.html'),
        NewsSource('CNN', 'http://rss.cnn.com/rss/cnn_allpolitics.rss', 'CNN_politics.html'),
        NewsSource('Los Angeles Times', 'http://www.latimes.com/local/political/rss2.0.xml', 'LAT_politics.html'),
        NewsSource('The Washington Post', 'http://feeds.washingtonpost.com/rss/rss_election-2012', 'TWP_politics.html'),
        NewsSource('The Denver Post', 'http://feeds.denverpost.com/dp-politics-national_politics', 'TDP_politics.html'),
    ]
    env['newsGroups']['business'] = [
        NewsSource('The Wall Street Journal', 'http://www.wsj.com/xml/rss/3_7031.xml', 'TWSJ_business.html'),
        NewsSource('BBC', 'http://feeds.bbci.co.uk/news/business/rss.xml', 'BBC_business.html'),
        NewsSource('CNN', 'http://rss.cnn.com/rss/money_latest.rss', 'CNN_business.html'),
        NewsSource('The New York Times', 'http://rss.nytimes.com/services/xml/rss/nyt/Economy.xml', 'TNYT_business.html'),
        NewsSource('USA Today', 'http://rssfeeds.usatoday.com/UsatodaycomMoney-TopStories', 'USAT_business.html'),
        NewsSource('Los Angeles Times', 'http://www.latimes.com/business/rss2.0.xml', 'LAT_business.html'),
        NewsSource('New York Times', 'https://nypost.com/business/feed/', 'NYT_business.html'),
    ]
    env['newsGroups']['technology'] = [
        NewsSource('BBC Technology', 'http://feeds.bbci.co.uk/news/technology/rss.xml', 'BBC_technology.html'),
        NewsSource('CNN Technology', 'http://rss.cnn.com/rss/cnn_tech.rss', 'CNN_technology.html'),
        NewsSource('New York Times Technology', 'http://feeds.nytimes.com/nyt/rss/Technology',  'NYT_technology.html'),
        NewsSource('NPR Technology', 'https://www.npr.org/rss/rss.php?id=1019', 'NPR_technology.html'),
    ]
    env['newsGroups'] = {'business': env['newsGroups']['business']}

# Get news items.
def DownloadNewsGroup(newsGroupName):
    newsItems = []
    for source in env['newsGroups'][newsGroupName]:
        if env['isCacheEnabled'] == False or source.filecache is None:
            Debug('Load from web for {0}'.format(source.description))
            rawNewsItems = feedparser.parse(source.link)
        else:
            if not os.path.isfile(source.filecache) or env['forceCacheRefresh'] == True:
                Debug('Update file cache for {0}'.format(source.description))
                content = urlopen(source.link).read().decode()
                open(source.filecache, 'w', encoding='utf-8').write(content)
            Debug('Load from file cache for {0}'.format(source.description))
            rawNewsItems = feedparser.parse(source.filecache)

        Debug('{0} news items loaded'.format(len(rawNewsItems.entries)))
        newsItems += [NewsItem(source, x) for x in rawNewsItems.entries]
    Debug('{0} news items loaded in total'.format(len(newsItems)))
    return newsItems

# Compute news graph.
def ComputeGraph(newsItems):
    # Compute NP nodes and NP edges. NP == keyword.
    Debug('Compute NP graph.')
    npNodes = OrderedDict()
    npEdges = set([])
    for i, item in enumerate(newsItems):
        npNodeIds = [] # NP node IDs of NPs in current newsItem.
        for np in item.nps: # Add nodes, record revert indices and NP node IDs.
            if np not in npNodes.keys():
                npNodes[np] = set([])
            npNodes[np].add(i) # Revert index: NP -> set(newsItem IDs).
            npNodeIds.append(list(npNodes.keys()).index(np)) # Get NP node ID.
        for j in range(len(npNodeIds)): # Add edges between NPs in a newsItem.
            for k in range(j + 1, len(npNodeIds)):
                npEdges.add((npNodeIds[j], npNodeIds[k]))

    # Rank NP nodes.
    Debug('Rank NP nodes.')
    sortedNPNodeIds = TextRank(len(npNodes), npEdges)

    ### TODO: refactor np-related names.
    ### TODO: revert lowercase words to original form.
    ### TODO: trump, president trump, ...
    ### TODO: news item order should matter?

    # Compute edges between NPs (aka keywords) and newsItems.
    Debug('Map keyword ID to newsItem IDs.')
    keywords = []
    keyword2newsItem = {} # Map keyword ID to a set of newsItem IDs.
    npNodeList = list(npNodes.items())
    for npNodeId in sortedNPNodeIds:
        keyword = npNodeList[npNodeId][0]
        newsItemList = npNodeList[npNodeId][1]
        keywords.append(keyword)
        keyword2newsItem[len(keywords) - 1] = set(newsItemList)
    return [keywords, keyword2newsItem]

    ''' Subgraph
        # Compute and sort pair-wise distance.
        s = len(itemArray)
        edges = pd.DataFrame(index=range(int(s*(s-1)/2)),columns=['i', 'j', 'distance'])

        ptr = 0
        for i in range(s):
            for j in range(i):
                dist = Distance(itemArray[i], itemArray[j])
                ### TODO: remove duplicate items (dist < epsilon)
                if dist > 1 - sys.float_info.epsilon:
                    continue
                edges.loc[ptr] = [i, j, Distance(itemArray[i], itemArray[j])]
                ptr += 1
        edges = edges[:ptr]
        dtypes = {'i': int, 'j': int, 'distance': np.float64}
        edges = edges.astype(dtypes)

        edges.sort_values(by=['distance'],inplace=True)
        edges = edges[:int(len(edges) * 0.2)]

        # Find disconnected subgraphs.
        subgraphs = FindSubgraphs(len(itemArray), edges)
        code.interact(local=locals())
    '''

    ''' For debugging distance.
import math
def D(i, j):
    itemI = itemArray[i]
    itemJ = itemArray[j]
    commonWords = itemI.words & itemJ.words
    return 1 - float(sum(itemI.hist[x] * itemJ.hist[x] for x in commonWords)) / (math.sqrt(sum(itemI.hist[x] ** 2 for x in itemI.words)) * math.sqrt(sum(itemJ.hist[x] ** 2 for x in itemJ.words)))

def I(i):
    print('URL: {0}\nTitle: {1}\nSummary: {2}'.format(itemArray[i].source, itemArray[i].title, itemArray[i].summary))

    '''

    ''' MDS
        itemDistMat = np.ndarray((len(itemArray), len(itemArray)), dtype=np.float64)
        for i in range(len(itemArray)): # Compute distance matrix.
            for j in range(i):
                itemDistMat[i, j] = itemDistMat[j, i] = Distance(itemArray[i], itemArray[j])
            itemDistMat[i, i] = 0.

        mds = MDS(n_components=3, dissimilarity='precomputed')
        coordinates = pd.DataFrame(mds.fit_transform(itemDistMat)) # Compute coordinates.

        # fig, ax = plt.subplots() # Visualize.
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(coordinates[0], coordinates[1], coordinates[2])
        for i, item in enumerate(itemArray):
            #ax.annotate(i, (coordinates.loc[i, 0], coordinates.loc[i, 1]))
            ax.text(coordinates.loc[i,0],coordinates.loc[i,1],coordinates.loc[i,2],  '%s' % (str(i)), size=10, zorder=1, color='k')
        plt.draw()
        plt.pause(0.001)
        for i in range(len(itemArray)):
            print(i, itemArray[i].title)
        code.interact(local=locals())
    '''

def GenHtml(newsItems, keywords, keyword2newsItem):
    code.interact(local=locals())
    '''
    res = open('index.html.template', 'r').read()
    nodeList = list(nodes.keys())
    thresh = 20
    nodesStr = ',\n'.join(["    {{id: {0}, label: '{1}'}}".format(i, html.escape(nodeList[nodeId])) for i, nodeId in enumerate(sortedNodeIds[:thresh])])
    edgesStr = ''
    res = res.replace('%NODES%', nodesStr, 1).replace('%EDGES%', edgesStr, 1)
    open('index.html', 'w').write(res)
    '''

def Msg(msg):
    print(msg)

def Debug(msg):
    if env['debug'] == True:
        Msg('[DEBUG] ' + msg)

def Error(msg):
    Msg('[ERROR] ' + msg)
    exit(1)




if __name__ == '__main__':
    LoadConfig()
    for newsGroupName in env['newsGroups'].keys():
        newsItems = DownloadNewsGroup(newsGroupName)
        keywords, keyword2newsItem = ComputeGraph(newsItems)
        GenHtml(newsItems, keywords, keyword2newsItem)
