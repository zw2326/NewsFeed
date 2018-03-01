# -*- coding: utf-8 -*
# https://simonsarris.com/making-html5-canvas-useful/
# https://en.wikipedia.org/wiki/List_of_newspapers_in_the_United_States
# http://michelleful.github.io/code-blog/2015/09/10/parsing-chinese-with-stanford/
from collections import defaultdict
from itertools import groupby
from rake import rake
from urllib.request import urlopen
import code
import datetime
import dateutil.parser
import feedparser
import html
import inspect
import math
import networkx
import os
import pytz
import re

# TODO:
# Custom tab template per news group.
# Subscribe to sports news group.
# Output config: HTML, JSON, SQL...

env = {}

# For RAKE.
raker = rake.Rake('rake/SmartStoplist.txt', 5, 3)

class NewsSource(object):
    def __init__(self, name, link, filecache=None, titleRe=None, summaryRe=None):
        self.name = name
        self.link = link
        self.filecache = filecache
        self.titleRe = titleRe
        self.summaryRe = summaryRe

class NewsItem(object):
    def __init__(self, source, entry):
        # Ordinary properties.
        self.link = entry.link
        self.thumbnails = entry.media_thumbnail if 'media_thumbnail' in entry.keys() else []
        self.source = source
        self.timestamp = entry.published if 'published' in entry.keys() else ''
        self.title = re.sub('<[^<]+?>', '', Translater.Translate(html.unescape(entry.title)))
        self.summary = re.sub('<[^<]+?>', '', Translater.Translate(html.unescape(entry.summary)))

        # Clean up title and summary.
        if source.titleRe is not None:
            self.title = source.titleRe(self.title)
        if source.summaryRe is not None:
            self.summary = source.summaryRe(self.summary)

        # Extract keywords from title and summary.
        self.kwt = [x[0] for x in raker.run(self.title)]
        self.kws = [x[0] for x in raker.run(self.summary)]
        self.keywords = set(self.kwt + self.kws) - env['blacklist']

        self.keywordsOrig = set([])
        for keyword in self.keywords:
            titleMatchIndex = self.title.lower().find(keyword)
            if titleMatchIndex != -1:
                self.keywordsOrig.add(self.title[titleMatchIndex:(titleMatchIndex + len(keyword))])
                continue
            summaryMatchIndex = self.summary.lower().find(keyword)
            if summaryMatchIndex != -1:
                self.keywordsOrig.add(self.summary[summaryMatchIndex:(summaryMatchIndex + len(keyword))])
                continue
            Error('Cannot locate original keyword "{0}" in title or summary'.format(keyword))

        # DEBUG
        # open('inspect.txt', 'a', encoding='utf-8').write('{0}\n{1}\n{2}\n==========\n{3}\n++++++++++\n{4}\n{5}\n==========\n\n'.format(self.source.name, self.title, self.summary, list(self.keywords), self.kwt, self.kws))
        # DEBUG

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
        '\u2014'   : '-',
    }

    utf2asciiPattern = '(' + '|'.join(chars.keys()) + ')'

    @classmethod
    def Translate(cls, content):
        return re.sub(Translater.utf2asciiPattern, lambda x: Translater.chars[x.group(0)], content)

# Load all configurations.
def LoadConfig():
    global env
    env = {
        'debug': True,
        'forceCacheRefresh': True,
        'isCacheEnabled': True,
        'newsGroups': {},
        'cacheDir': 'cache',
        'templateFile': 'index.template',
        'outputFile': 'WEB-INF/index.html',
    }
    newsGroups = defaultdict(list) # Load from config file and overwrite env['newsGroups'].
    blacklist = set([]) # Load from config file and overwrite env['blacklist']

    # Load config file.
    configFile = 'config.ini'
    if os.path.isfile(configFile):
        section = None
        newsGroup = None
        fid = open(configFile, 'r')
        for line in fid:
            line = line.strip()
            if line == '' or line.startswith('#'): # Blank or comment.
                continue

            if re.match('^=====.*=====$', line) is not None: # Section line.
                section = re.search('^===== *([^ ]*) *=====$', line).groups(0)[0]
                if section.startswith('newsgroup'): # Get newsgroup name for newsgroup section.
                    newsGroup = re.search('^newsgroup-(.*)', section).groups(0)[0]
                    section = 'newsgroup'
                continue

            if section == 'config':
                key, value = [x.strip() for x in re.split('=' , line, 1)]
                if key in env.keys() and type(env[key]) == bool:
                    env[key] = bool(value)
                else:
                    env[key] = value
            elif section == 'newsgroup':
                values = re.split(' {2,}|\t', line, 3)
                # Generate cache filename by removing punc (except hyphen) and space from descriptive name and concat with newsgroup,
                # e.g. New York Times -> NewYorkTimes_politics.html
                cache = '{0}_{1}.html'.format(re.sub(r'[^a-zA-Z0-9\-]+', '', values[0]), newsGroup)
                values.insert(2, cache)
                # Parse lambda functions for title and summary.
                for i in range(3, len(values)):
                    exec('values[i] = ' + values[i]) # Dangerous work around!
                newsGroups[newsGroup].append(NewsSource(*values))
            elif section == 'blacklist':
                blacklist.add(line)

    env['newsGroups'] = newsGroups
    env['blacklist'] = blacklist

    if not os.path.isdir(env['cacheDir']):
        os.makedirs(env['cacheDir'])

# Get news items from each source in a news group.
def DownloadNewsGroup(newsGroupName):
    newsItems = []
    timeParseErrors = defaultdict(list)
    for source in env['newsGroups'][newsGroupName]:
        rawNewsItems = DownloadRawNewsItems(source)
        newsItems += ConvertNewsItems(source, rawNewsItems, timeParseErrors)
    Debug('{0} news items loaded in total'.format(len(newsItems)))

    # Generate report for timestamp parsing errors.
    msg = 'Timestamp parsing errors:\n'
    for k, v in timeParseErrors.items():
        msg += '{0}:'.format(k)
        for value, repeated in groupby(sorted(v)):
            msg += ' {0}({1})'.format(value, sum(1 for _ in repeated))
    Debug(msg.strip())
    return newsItems

def DownloadRawNewsItems(source):
    rawNewsItems = None
    if env['isCacheEnabled'] == False or source.filecache is None:
        Debug('Load from web for {0}'.format(source.name))
        rawNewsItems = feedparser.parse(source.link)
    else:
        cacheFile = os.path.join(env['cacheDir'], source.filecache)
        if not os.path.isfile(cacheFile) or env['forceCacheRefresh'] == True:
            Debug('Update file cache for {0}'.format(source.name))
            try:
                content = urlopen(source.link).read().decode()
            except Exception as e:
                Msg("Update file cache for {0} failed: {1}".format(source.name, e))
                return rawNewsItems
            open(cacheFile, 'w', encoding='utf-8').write(content)
        Debug('Load from file cache for {0}'.format(source.name))
        rawNewsItems = feedparser.parse(cacheFile)
    return rawNewsItems

def ConvertNewsItems(source, rawNewsItems, timeParseErrors):
    if rawNewsItems is None:
        return []

    utcNow = datetime.datetime.now(datetime.timezone.utc)
    newsItems = []
    count = 0
    for rawNewsItem in rawNewsItems.entries:
        newsItem = NewsItem(source, rawNewsItem)

        # Discard news items without valid timestamp or older than 2 days.
        timestamp = None
        try:
            timestamp = dateutil.parser.parse(newsItem.timestamp)
            if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
                # Use UTC if timestamp is naive.
                timestamp = timestamp.replace(tzinfo=pytz.utc)
        except Exception as e:
            # Silently pass through if timestamp cannot be parsed.
            timeParseErrors[e.__str__()].append(newsItem.source.name)
            continue
        if timestamp is None or (utcNow - timestamp).total_seconds() >= 86400 * 2:
            continue

        newsItems.append(newsItem)
        count += 1

    Debug('{0} news items loaded, {1} accepted'.format(len(rawNewsItems.entries), count))
    Msg('[{0}] {1}'.format('OK' if count > 0 else 'FAIL', source.name))
    return newsItems

# Compute news graph.
def ComputeGraph(newsItems):
    # Compute newsItem graph.
    Debug('Compute news item graph.')
    edges = set([])
    for i in range(len(newsItems)):
        for j in range(i + 1, len(newsItems)):
            itemI = newsItems[i]
            itemJ = newsItems[j]
            commonKeywords = itemI.keywords & itemJ.keywords
            if len(commonKeywords) == 0:
                continue
            weight = len(commonKeywords) / math.sqrt(len(itemI.keywords)*len(itemJ.keywords))
            edges.add((i, j, weight))

    # Rank news items.
    Debug('Rank news Items.')
    sortedNewsItemIds = TextRank(len(newsItems), edges)

    # Sort news items.
    Debug('Sort news items.')
    sortedNewsItems = []
    for i in sortedNewsItemIds:
        sortedNewsItems.append(newsItems[i])
    return sortedNewsItems

def GenHtml(newsItems):
    Debug('Write to html.')
    res = open(env['templateFile'], 'r').read()
    res = res.replace('%CATEGORY_LIST%', '\n'.join(['        <li{0}><a href="#{1}-tab" data-toggle="tab">{1}</a></li>'.format(
            ' class="active"' if i == 0 else '', x) for i, x in enumerate(env['newsGroups'].keys())]))

    # NewsItem string templates.
    newsItemStrTemp = '''
        <div class="newsitem">
{IMG}
          <h5 class="title"><a href="{NEWS_URL}"><b>{TITLE}</b></a></h5>
          <p class="source"><a href="{SOURCE_URL}">{SOURCE}</a></p>
          <p class="timestamp">{TIMESTAMP}</p>
          <p class="keywords">{KEYWORDS}</p>
          <p class="summary">{SUMMARY}</p>
        </div>'''
    newsItemImgStrTemp = '''          <img class="thumbnail" src="{THUMBNAIL_URL}" width="{WIDTH}"/>'''
    newsItemKeywordStrTemp = '''<p class="keyword">{KEYWORD}</p>'''

    newsTabStrs = []
    for i, newsGroupName in enumerate(env['newsGroups'].keys()):
        newsItemStrs = []
        for newsItem in newsItems[newsGroupName]:
            # Prepare thumbnail.
            if newsItem.thumbnails == []:
                newsItemImgStr = ''
            else:
                if 'width' not in newsItem.thumbnails[0].keys():
                    width = 300
                else:
                    width = min(300, int(newsItem.thumbnails[0]['width']))
                newsItemImgStr = newsItemImgStrTemp.format(THUMBNAIL_URL=newsItem.thumbnails[0]['url'], WIDTH=width)

            # Prepare keywords.
            keywordsStr = ''.join(newsItemKeywordStrTemp.format(KEYWORD=html.escape(x)) for x in sorted(newsItem.keywordsOrig, key=lambda s: s.lower()))

            newsItemStrs.append(newsItemStrTemp.format(
                IMG=newsItemImgStr,
                NEWS_URL=newsItem.link,
                TITLE=html.escape(newsItem.title),
                SOURCE_URL=newsItem.source.link,
                SOURCE=html.escape(newsItem.source.name),
                TIMESTAMP=html.escape(newsItem.timestamp),
                KEYWORDS=keywordsStr,
                SUMMARY=html.escape(newsItem.summary),
            ))
        newsTabStr = '''
      <div class="tab-pane category-tab{0}" id="{1}-tab">
{2}
      </div>'''.format(" active" if i == 0 else "", newsGroupName, '\n'.join(newsItemStrs))
        newsTabStrs.append(newsTabStr)

    res = res.replace('%NEWS_TABS%', '\n'.join(newsTabStrs))
    open(env['outputFile'], 'w', encoding='utf-8').write(res)

def TextRank(n, edges):
    # Build graph.
    graph = networkx.Graph()
    graph.add_nodes_from(range(n)) # Add nodes.
    graph.add_weighted_edges_from(edges) # Add deges.

    # Rank nodes.
    ranks = networkx.pagerank(graph) # Page rank.
    sortedNodes = [x[0] for x in sorted(ranks.items(), key=lambda x: x[1], reverse=True)]
    return sortedNodes

# Filter news items by keyword.
def Filter(keyword, newsItems, notInclude=False):
    filteredNewsItems = []
    for i in newsItems:
        if notInclude == any([x.find(keyword) != -1 for x in i.keywords]):
            continue
        filteredNewsItems.append(i)
    return filteredNewsItems

def Msg(msg):
    print(msg)

def Debug(msg):
    callerDebugID = 'debug_{0}'.format(inspect.stack()[1][3])
    if env['debug'] == True and (callerDebugID not in env.keys() or (callerDebugID in env.keys() and env[callerDebugID] == 'True')):
        Msg('[DEBUG] ' + msg)

def Error(msg):
    Msg('[ERROR] ' + msg)
    exit(1)




if __name__ == '__main__':
    LoadConfig()
    sortedNewsItems = {}
    for newsGroupName in env['newsGroups'].keys():
        Debug('Category {0} started.'.format(newsGroupName))
        newsItems = DownloadNewsGroup(newsGroupName)
        sortedNewsItems[newsGroupName] = ComputeGraph(newsItems)
        Debug('Category {0} finished.\n'.format(newsGroupName))
    GenHtml(sortedNewsItems)
