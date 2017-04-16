import numpy as np
import data
from utils import *


def main():
    train, test = data.get_train_test()
   
    metadata = []
    photo = []
    uid = []
    for img in train:
        pictureID = img['watchlink'].strip()
        tags = img['tags']
        lat, lon = img['latitude'], img['longitude']
        userID = img['userID']

        metadata.append('{0}\t0\t0\t{1}'.format(pictureID, ', '.join(tags)))
        photo.append('0\t0\t{0}\t{1}'.format(lon, lat))
        uid.append('0\t{0}'.format(userID))

    write_to_file(metadata, 'train_metadata')
    write_to_file(photo, 'train_photo')
    write_to_file(uid, 'train_uid')


def remove_test_data(all_data):
    ca_all = data.filter(all_data, 'ca')
    ca_train, ca_test = data.split(ca_all, 0.8)
    ca_test_ids = set()
    for entry in ca_test:
        ca_test_ids.add(entry['watchlink'])

    return [x for x in all_data if x['watchlink'] not in ca_test_ids]

def write_to_file(lst, name):
    f = open(name, 'w')
    for item in lst:
        f.write(item + '\n')
    f.close()

if __name__ == '__main__':
    main()

