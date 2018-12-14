import os
import json

def count_ids(root, flag=0):
    ids_dict = {}
    captions = 0
    with open(root,'r') as f:
        info = json.load(f)
        for data in info:
            label = data['id'] - flag
            ids_dict[label] = ids_dict.get(label,0) + 1
            captions += len(data['captions'])
    return ids_dict, captions


if __name__ == "__main__":
    data_root = '/home/zhangqi/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching/data'
    total_statistics, captions_total = count_ids(os.path.join(data_root, 'reid_raw.json'),flag=1)
    print('{}-id {}-images {}-captions in total'.format(len(total_statistics), sum(total_statistics.values()), captions_total))
    processed_root = os.path.join(data_root, 'processed_data')
    train_statistics, captions_train = count_ids(os.path.join(processed_root, 'train_reid.json'))
    print('{}-id {}-images {}-captions in train'.format(len(train_statistics), sum(train_statistics.values()), captions_train))
    val_statistics, captions_val = count_ids(os.path.join(processed_root, 'val_reid.json'))
    print('{}-id {}-images {}-captions in val'.format(len(val_statistics), sum(val_statistics.values()), captions_val))
    test_statistics, captions_test = count_ids(os.path.join(processed_root, 'test_reid.json'))
    print('{}-id {}-images {}-captions in test'.format(len(test_statistics), sum(test_statistics.values()), captions_test))
    for index, (key, total_num) in enumerate(total_statistics.items()):
        if key in train_statistics:
            if total_num == train_statistics[key]:
                continue
            else:
                print(key)
                print(total_num - train_statistics[key])
        if key in val_statistics:
            if total_num == val_statistics[key]:
                continue
            else:
                print(key)
                print(total_num - val_statistics[key])
        if key in test_statistics:
            if total_num == test_statistics[key]:
                continue
            else:
                print(key)
                print(total_num - val_statistics[key])
        print('-------------------{}---------------------'.format(index))

