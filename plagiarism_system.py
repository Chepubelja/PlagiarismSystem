from os import listdir
from string import punctuation
from time import time
import numpy as np
from sklearn.metrics import jaccard_similarity_score


class PlagiarismSystem(object):
    """
    Plagiarism System Class.
    """
    def __init__(self, shingle_len, folder_dir, num_permutations, similarity_plagiarism):
        """
        Class Constructor.
        """
        self.shingle_len = shingle_len
        self.folder_dir = folder_dir
        self.num_permutations = num_permutations
        self.similarity_plagiarism = similarity_plagiarism
        # ==========================================
        self.all_doc_shingles, self.unique_shingles = self.shingle_all_files()
        self.signature_matrix = self.create_signatures()
        self.check_plagiarism()

    def shingle_all_files(self):
        """
        Function that read all txt files  and divide all text on number of mappers.
        :return:
        unique_shingles: Dictionary with all unique shingles,
            where key is a shingle and value of this key is its index.
        all_doc_shingles: Dictionary with all shingles indices that are in each document, where
            key is the document name and value set of all shingles that appears in this document.
        """
        shingle_id, all_doc_shingles, unique_shingles = 0, {}, {}

        for filename in listdir(self.folder_dir):
            with open(self.folder_dir + filename, encoding='latin-1') as file:
                data = self.remove_punctuation(file.read().replace('\n', '')
                                               .replace('â\x80\x98', '').
                                               replace('â\x80\x99', '')).split()
                doc_shingles = []
                for i in range(len(data) - self.shingle_len + 1):
                    shingle = self.get_shingle(data, i)
                    if shingle not in unique_shingles:
                        shingle_id += 1
                        unique_shingles[shingle] = shingle_id
                    doc_shingles.append(unique_shingles[shingle])
                all_doc_shingles[filename] = set(doc_shingles)

        return all_doc_shingles, unique_shingles

    def get_shingle(self, text, index):
        """
        Returns single shingle.
        """
        return " ".join([word for word in text[index:index + self.shingle_len]])

    def check_plagiarism(self):
        """
        Checks for plagiarism between documents and prints all
        document-document pairs that are plagiarism.
        """
        plagiarism = 0
        for i in range(len(self.signature_matrix)):
            for j in range(i, len(self.signature_matrix)):
                if i == j:
                    pass
                else:
                    similarity = jaccard_similarity_score(self.signature_matrix[i],
                                                          self.signature_matrix[j])
                    if similarity >= self.similarity_plagiarism:
                        plagiarism += 1
                        print('Files ' + list(self.all_doc_shingles)[i] +
                              ' and ' + list(self.all_doc_shingles)[j],
                              'are plagiarism on:', round(similarity * 100, 2), '%')
        print()
        print('Summary: ' + str(plagiarism) + ' files with plagiarism.')

    def min_hash(self):
        """
        Finds the first non-zero value in each column.
        :return: List of shape (number of documents, 1), where each element
            is the index of first occurence of shingle in the document.
        """
        min_values = np.empty(len(listdir(self.folder_dir)))

        shuffled_rows = np.random.permutation(len(self.unique_shingles))

        for j, doc in enumerate(listdir(self.folder_dir)):
            for i, shingle_index in enumerate(shuffled_rows):
                if shingle_index in self.all_doc_shingles[doc]:
                    min_values[j] = i + 1
                    break
        return min_values

    def create_signatures(self):
        """
        Creates signature matrix.
        :return: Signature matrix with shape (number of documents, number of permutations)
        """
        signature_matrix = np.empty((len(self.all_doc_shingles), self.num_permutations))

        for i in range(self.num_permutations):
            signature_matrix[0:(len(self.all_doc_shingles)), i] = self.min_hash()
        return signature_matrix

    @staticmethod
    def remove_punctuation(input_str):
        """
        Removes all punctuation from string.
        :param input_str: Input string.
        :return: String without any punctuation.
        """
        return ''.join(ch for ch in input_str if ch not in set(punctuation)) + ' '


def main():
    """
    Main function.
    """
    start_time = time()

    PlagiarismSystem(shingle_len=3, folder_dir="corpus-20090418/",
                     num_permutations=200, similarity_plagiarism=0.33)

    print("Time of execution:", time() - start_time, "sec")


if __name__ == "__main__":
    main()
