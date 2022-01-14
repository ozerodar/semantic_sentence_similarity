import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import random
import re
import wget

from itertools import combinations
from experiments.utils import get_json, save_to_json, DATA_PATH, get_datasets

random.seed(1)


def get_entities(dataset):
    entities = get_json("entities_{}.json".format(dataset))
    return [sorted([syn.lower() for syn in synonyms], key=len, reverse=True) for synonyms in entities]


def get_score(dataset, pair):
    for x in dataset:
        if x[0] == pair[0] and x[1] == pair[1] or x[0] == pair[1] and x[1] == pair[0]:
            return x[2]
    return None


class DatasetCreator:
    def __init__(
        self,
        dataset="harry_potter",
        train_file=None,
        dev_file=None,
        test_file=None,
        train_ratio=0.7,
        dev_ratio=0.15,
        num_samples=20,
        max_entities=8,
    ):
        self.dataset_name = dataset
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.entities = get_entities(self.dataset_name)
        self.intents = get_json("intents_{}.json".format(self.dataset_name))
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = 1 - train_ratio - dev_ratio
        self.num_samples = num_samples
        self.max_entities = max_entities

        self.stop_words = []
        for entity in self.entities:
            for syn in entity:
                self.stop_words.append(syn)

        self.keyboard_aug = nac.KeyboardAug(
            aug_char_p=0.1,
            aug_word_p=0.1,
            aug_word_max=2,
            include_special_char=False,
            include_numeric=False,
            include_upper_case=False,
            stopwords=self.stop_words,
            verbose=0,
        )
        self.swap_aug = nac.RandomCharAug(
            action="swap",
            aug_char_p=0.1,
            aug_word_p=0.1,
            aug_word_max=2,
            include_numeric=False,
            include_upper_case=False,
            stopwords=self.stop_words,
            verbose=0,
        )

        path = DATA_PATH / "spelling_en.txt"
        if not path.exists():
            wget.download(
                "https://raw.githubusercontent.com/makcedward/nlpaug/5238e0be734841b69651d2043df535d78a8cc594/nlpaug/res/word/spelling/spelling_en.txt",
                out="../data/spelling_en.txt",
            )
        else:
            print("File already exists")
        self.err_aug = naw.SpellingAug(dict_path=path)

    def divide_data(self):
        data = {"train": [], "dev": [], "test": []}

        ctr = 0
        sentences = []
        for intent in self.intents:
            sentences.extend(list(intent.values())[0])
        n = len(sentences)
        ctr += n
        n_train = int(self.train_ratio * n)
        n_dev = int(self.dev_ratio * n)
        n_test = int(self.test_ratio * n)
        n_train += n - n_train - n_dev - n_test

        train = [sentences.pop(random.randrange(len(sentences))).lower() for _ in range(n_train)]
        dev = [sentences.pop(random.randrange(len(sentences))).lower() for _ in range(n_dev)]
        test = [sentences.pop(random.randrange(len(sentences))).lower() for _ in range(n_test)]

        # print(f"trn: {train}, dev: {dev}, tst: {test}")
        data["train"].extend(train)
        data["dev"].extend(dev)
        data["test"].extend(test)
        print(f"sentences: {ctr}")
        return data

    def create_scored_samples(self, sentences, rate):
        scored_dataset = get_json("scored_{}.json".format(self.dataset_name))  # TODO: name to the class var
        data = []
        ctr = 0

        if len(sentences) > 1:
            comb = list(combinations(sentences, 2))
            # pairs = random.sample(comb, self.num_samples)
            while ctr < self.num_samples * rate:
                pair = random.sample(comb, 1)[0]
                # for pair in pairs:
                if score := get_score(scored_dataset, pair) is None:
                    continue
                    # user_input = input(f"Input similarity score for a pair. 's' to skip, 'x' save and quit\n{pair}\n")
                    # if user_input == "x":
                    #     break
                    # elif user_input == "s":
                    #     continue
                    # try:
                    #     score = float(user_input)
                    #     scored_dataset.append([pair[0], pair[1], score])
                    # except ValueError:
                    #     continue
                data.append([pair[0], pair[1], float(score)])
                ctr += 1
            save_to_json("scored_{}.json".format(self.dataset_name), scored_dataset)
        return data

    def entities_in_sentence(self, sentence):
        entities = {}
        for synonyms in self.entities:
            for syn in synonyms:
                regex = re.compile(r"\b%s\b" % syn, re.I)
                if regex.search(sentence):
                    entities[syn] = synonyms
                    break
        return entities

    def augment_synonyms(self, sentences):
        pairs = []
        counter = {synonyms[0]: 0 for synonyms in self.entities}

        for sent in sentences:
            sentence = sent.lower()

            if synonyms := self.entities_in_sentence(sentence):
                for ent, syn in synonyms.items():
                    if counter[syn[0]] < self.max_entities:
                        for i in range(len(syn) - 1):
                            pairs.append([sentence.replace(ent, syn[i]), sentence.replace(ent, syn[i + 1]), 1.0])
                        counter[syn[0]] += 1
        return pairs

    def augment_typos(self, data):
        data_with_typos = []

        for sample in data:
            pair0_typo = self.keyboard_aug.augment(sample[0])
            pair1_typo = self.keyboard_aug.augment(sample[1])
            pair0_swap = self.swap_aug.augment(pair0_typo)
            pair1_swap = self.swap_aug.augment(pair1_typo)

            data_with_typos.append([pair0_swap, pair1_swap, sample[2]])
        return data_with_typos

    def augment_grammar(self, data):
        data_with_errors = []

        for sample in data:
            # omit the articles
            # pair0_error = re.sub(r'(\b|\s)the(\s|\b)', ' ', sample[0])
            # pair0_error = re.sub(r'(\b|\s)a(\s|\b)', ' ', pair0_error).strip()
            # pair1_error = re.sub(r'(\b|\s)the(\s|\b)', ' ', sample[1])
            # pair1_error = re.sub(r'(\b|\s)a(\s|\b)', ' ', pair1_error).strip()

            pair0_error = self.err_aug.augment(sample[0])
            pair1_error = self.err_aug.augment(sample[1])

            data_with_errors.append([pair0_error, pair1_error, sample[2]])
        return data_with_errors

    def augment(self, sentences, rate):
        data = []
        data.extend(self.augment_synonyms(sentences))
        data.extend(self.create_scored_samples(sentences, rate))
        return data

    def create(self, update=True, max_samples=100):
        x_trn, y_trn, x_dev, y_dev, x_tst, y_tst = get_datasets(
            train=self.train_file, dev=self.dev_file, test=self.test_file
        )
        dataset = {} if not update else self.divide_data()

        for x, y, split in zip([x_trn, x_dev, x_tst], [y_trn, y_dev, y_tst], ["train", "dev", "test"]):
            if not update:
                dataset[split] = [[pair[0], pair[1], score] for pair, score in zip(x[0:max_samples], y[0:max_samples])]
            else:
                rate = self.train_ratio if split == "train" else self.test_ratio
                dataset[split] = self.augment(dataset[split], rate)
            save_to_json("STS_{}_2021_{}.json".format(split, self.dataset_name), dataset[split])

            dataset_with_typos = self.augment_typos(dataset[split])
            dataset_with_errors = self.augment_grammar(dataset[split])
            dataset_with_errors_and_typos = self.augment_typos(dataset_with_errors)
            save_to_json("STS_{}_2021_{}_typos.json".format(split, self.dataset_name), dataset_with_typos)
            save_to_json("STS_{}_2021_{}_errors.json".format(split, self.dataset_name), dataset_with_errors)
            save_to_json(
                "STS_{}_2021_{}_errors_typos.json".format(split, self.dataset_name), dataset_with_errors_and_typos
            )


if __name__ == "__main__":
    stsb_dataset = DatasetCreator(
        train_file="stsbenchmark.tsv.gz",
        dev_file="stsbenchmark.tsv.gz",
        test_file="stsbenchmark.tsv.gz",
        dataset="stsb",
    )
    stsb_dataset.create(update=False, max_samples=100000)

    harry_potter_dataset = DatasetCreator(num_samples=500)
    harry_potter_dataset.create(update=True)
