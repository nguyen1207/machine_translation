import html
import multiprocessing as mp
import random
import re
import string
from threading import Thread

import nltk
import numpy as np
import pandas as pd
from datasets import load_dataset
from nltk.corpus import stopwords
from tqdm import tqdm
from underthesea import word_tokenize
from wordsegment import load, segment

nltk.download("stopwords")


vi_stopwords = set()
en_stopwords = set(stopwords.words("english"))

with open("./data/stopwords.txt", "r", encoding="utf8") as file:
    for line in file:
        vi_stopwords.add(line.strip())


def remove_vi_stopwords(text):
    return " ".join([word for word in text.split(" ") if word not in vi_stopwords])


def remove_en_stopwords(text):
    return " ".join([word for word in text.split(" ") if word not in en_stopwords])


np.random.seed(42)
load()


non_teeencode_seed = load_dataset("mt_eng_vietnamese", "iwslt2015-vi-en")

lines = []

html_escape_table = {
    " &amp;": "&",
    " &quot;": '"',
    " &apos;": "'",
    " &gt;": ">",
    " &lt;": "<",
}


def html_unescape(text):
    for k, v in html_escape_table.items():
        text = re.sub(k, v, text)
    return text


def process(pair):
    vi = word_tokenize(html.unescape(pair["vi"]), format="text")
    en = html_unescape(pair["en"])

    for c in string.punctuation:
        if c == "_":
            continue

        if c in ('"', "'"):
            regex = rf"(^{c}|\s{c}|{c}\s|{c}$)"
            vi = re.sub(regex, "", vi)
            en = re.sub(regex, "", en)
        else:
            vi = re.sub(re.escape(str(c)), "", vi)
            en = re.sub(re.escape(str(c)), "", en)

    vi = vi.strip()
    en = en.strip()

    if len(vi) == 0 or len(en) == 0:
        return None

    line = f"{vi} ||| {en}\n"

    return line.lower()


threads = []

MAX_WORKERS = 20
CHUNK_SIZE = 1


def main():
    with mp.Pool(processes=MAX_WORKERS) as pool:
        inputs = non_teeencode_seed["train"]["translation"]

        results = tqdm(
            pool.imap_unordered(process, inputs, chunksize=CHUNK_SIZE),
            total=len(inputs),
        )

        with open("./src/text.vi-en", "a", encoding="utf8") as outfile:
            for result in results:
                if result is not None:
                    outfile.write(result)


if __name__ == "__main__":
    main()
