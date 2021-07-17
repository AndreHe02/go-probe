from annotated_datasets import *
from io_utils import *

def main():
    go_dict = read_pkl('data/sorted_go_dict.pkl')
    keywords = go_dict[:30]
    print('writing elf features and labels for keywords:')
    print(keywords)

    dataset = get_annotated_dataset('data/filtered_annotations.pkl', AGZ_features, better_bag_of_words(keywords))
    save_dataset(dataset, 'data/elf_inputs/', progress_bar=True)

if __name__ == '__main__':
    main()
