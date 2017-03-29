import numpy as np
import data
from utils import *

def main():
    all_data = data.get()
    
    metadata = []
    photo = []
    uid = []
    for img in all_data:
        pictureID = img['watchlink'].strip()
        tags = img['tags']
        lat, lon = img['latitude'], img['longitude']
        userID = img['userID']

        metadata.append('{0}\t0\t0\t{1}'.format(pictureID, ', '.join(tags)))
        photo.append('0\t0\t{0}\t{1}'.format(lon, lat))
        uid.append('0\t{0}'.format(userID))

    write_to_file(metadata, 'large_train_metadata')
    write_to_file(photo, 'large_placing_train_photo')
    write_to_file(uid, 'large_train_uid')

def write_to_file(lst, name):
    f = open(name, 'w')
    for item in lst:
        f.write(item + '\n')
    f.close()

if __name__ == '__main__':
    main()

