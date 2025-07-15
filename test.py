from dataset.Babilong import BABILongDataset
from dataset.Wikitext import WikiTextDataset

def test_dataset(tag, max_items):
    counter = 0
    if tag == "babilong":
        data = BABILongDataset("./dataset/babilong/8k")
    elif tag == "wikitext":
        data = WikiTextDataset("./dataset/wikitext")
    print("[INFO] End of creating of {} dataset ...".format(tag))
    dataLoader = data.get_data_loader()
    print("[INFO] End of getting data loader ...")
    for item in dataLoader:
        counter += 1
        print("[INFO] input_ids: {}".format(item['input_ids'].shape))
        print("[INFO] attention_mask: {}".format(item['attention_mask'].shape))
        print("[INFO] labels: {}".format(item['labels'].shape))
        if counter == max_items:
            break
    print('[INFO] Success ...')


if __name__ == "__main__":
    # test_dataset("babilong", 100)
    test_dataset("wikitext", 100)