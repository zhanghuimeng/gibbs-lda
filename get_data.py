import urllib.request
from pyquery import PyQuery as pq
from nltk.tokenize import TweetTokenizer
import string


ICLR_18 = "https://raw.githubusercontent.com/zhanghuimeng/gibbs-lda/master/raw_data/ICLR%202018%20Conference%20_%20OpenReview.html"
ICLR_19 = "https://raw.githubusercontent.com/zhanghuimeng/gibbs-lda/master/raw_data/ICLR%202019%20Conference%20_%20OpenReview.html"
URL_LIST = [ICLR_18, ICLR_19]


def parse_website_1819(d, year):
    papers = []

    for li_d in d("#accepted-poster-papers ul li.note").items():
        # print(type(li_d))
        title = li_d("h4 a:first").text()
        colp_d = li_d(".collapse .note-contents-collapse ul li")
        abstract = ""
        keywords = ""
        tldr = ""
        i = 0
        for in_d in colp_d.items():
            if i == 0:
                abstract = in_d(".note-content-value").text()
            elif i == 1:
                keywords = in_d(".note-content-value").text()
            elif i == 2:
                tldr = in_d(".note-content-value").text()
            i += 1
        papers.append({
            "title": title, 
            "abstract": abstract,
            "keywords": keywords,
            "tldr": tldr,
            "year": year,
        })
        # print("Title: " + title)
        # print("ABS: " + abstract)
        # print("KEY: " + keywords)
        # print("TL;DR:" + tldr)

    print("%d year paper number: %d" % (year, len(papers)))

    return papers


papers = []

for url in URL_LIST:
    print("Processing URL %s" % url)
    data = urllib.request.urlopen(url).read()
    print("Size of html file: %d" % len(data))

    d = pq(data)
    if url == ICLR_19:
        papers += parse_website_1819(d, 2019)
    elif url == ICLR_18:
        papers += parse_website_1819(d, 2018)

with open("data.txt", "w") as f:
    tknzr = TweetTokenizer(preserve_case=False)
    for paper in papers:
        tokens = tknzr.tokenize(paper["title"])
        tokens = list(filter(lambda token: token not in string.punctuation, tokens))
        f.write(" ".join(tokens) + "\n")
