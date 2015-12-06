import json
import re

from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from snowballstemmer import RussianStemmer, EnglishStemmer


def create_compiled_url_regexp():
    regexp = r'(?:http(?:s)?:\\/\\/)?(?:(?:[a-zа-я]|[a-zа-я0-9_-]{2,})\\.)+\
    (?:[a-z][a-z0-9-]{1,20}|рф)(?:\\/[!a-z0-9а-яё_z%~:\\.,-]*)*\
    (?:[\\?&#+][a-z0-9\\[\\]_]*(?:=?[a-z0-9~\\._=,%\\|+!-]*))*(?<![\\.,:!?-])'
    return re.compile(regexp)


def create_compiled_url_regexp_2():
    regexp = r'(\b(https?|ftp|file)://)?\
    [-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    return re.compile(regexp)


url_regexp = create_compiled_url_regexp()
url_regexp2 = create_compiled_url_regexp_2()


def replace_urls(text: str) -> str:
    text = url_regexp.sub(' ', text)
    text = url_regexp2.sub(' ', text)
    return text


def delete_non_word_chars(text: str) -> str:
    temp = replace_urls(text)
    temp = temp.replace('ё', 'е')
    temp = re.sub(r'(&[a-z0-9]*;)', ' ', temp)  # & encoded symbols
    temp = re.sub(r'(\W|\d)+', ' ', temp)  # non word or digit
    temp = re.sub(r'\s+', ' ', temp)  # 2+ spaces
    return temp.strip()


def split_text(text: str) -> list:
    return regexp_tokenize(text, '''[\w']+''')


def filter_variable_names(words: list) -> list:
    return [word for word in words if '_' not in word]


russian_stops = stopwords.words('russian')
english_stops = stopwords.words('english')


def filter_stopwords(words: list) -> list:
    return [word for word in words
            if word not in russian_stops and word not in english_stops]


russian_stemmer = RussianStemmer()
english_stemmer = EnglishStemmer()


def stem_words(words: list) -> list:
    stemmed = russian_stemmer.stemWords(words)
    stemmed = english_stemmer.stemWords(stemmed)
    return stemmed


def filter_words_with_repeatable_letters(words: list) -> list:
    return [word for word in words if not re.match('(.)\\1{2}', word)]


def is_language_usual_word(word: str) -> bool:
    length = len(word)
    is_eng = re.match('[a-z]', word)
    return length > 2 and \
           ((not is_eng and length < 25) or (is_eng and length < 15))


def filter_words_with_unusual_by_language_length(words: list) -> list:
    return [word for word in words if is_language_usual_word(word)]


def tokenize_text(text: str) -> list:
    text = text.lower()
    text = delete_non_word_chars(text)
    tokens = split_text(text)
    tokens = filter_variable_names(tokens)
    tokens = filter_stopwords(tokens)
    tokens = stem_words(tokens)
    tokens = filter_words_with_repeatable_letters(tokens)
    tokens = filter_words_with_unusual_by_language_length(tokens)
    return tokens


def normalize_label(label: str) -> str:
    label = label.lower()
    label = label.replace('ё', 'е')
    label = split_text(label)
    label = russian_stemmer.stemWords(label)
    label = english_stemmer.stemWords(label)
    return ' '.join(label)


label_black_list = {'лог компании', 'рная дыра', 'пиарюсь'}


def filter_labels_from_black_list(labels: list) -> list:
    return [label for label in labels if
            all(label_bl not in label for label_bl in label_black_list)]


def normalize_labels(labels: list) -> list:
    return [normalize_label(label) for label
            in filter_labels_from_black_list(labels)]


def normalize_tags(tags: list) -> list:
    return normalize_labels(tags)


def transform_data(data: dict) -> dict:
    hubs = normalize_labels(data['Hubs'])
    tags = normalize_tags(data['Tags'])
    tokens = tokenize_text(data['Text'])
    return {'Id': data['Number'],
            'Labels': list(set(hubs + tags)),
            'Features': tokens}


raw_data = sc.textFile('hdfs://master:54310/raw_data')
data = raw_data.map(json.loads)
transformed_data = data.map(lambda x: transform_data(x)) \
    .filter(lambda x: x['Features'] and x['Labels'])
transformed_data.map(lambda x: json.dumps(x)) \
    .repartition(6) \
    .saveAsTextFile('hdfs://master:54310/exp_f/clean_5+')
