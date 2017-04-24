import argparse

import numpy as np

import data
from utils import *


def main(file_size):
    data_funcs_by_size = {'small': data.get_small, 'medium': data.get_medium, 'large': data.get_large}
    all_data = data_funcs_by_size[file_size]()
    train, test = data.split(all_data, 0.8)
   
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

    write_to_file(metadata, file_size + '_train_metadata')
    write_to_file(photo, file_size + '_train_photo')
    write_to_file(uid, file_size + '_train_uid')


def write_to_file(lst, name):
    f = open(name, 'w')
    for item in lst:
        f.write(item + '\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--maxiter', nargs=1, type=int,
            help='Max number of iterations to run.')
    parser.add_argument('--small', action='store_const', const=1,
            help='Use a large dataset.')
    parser.add_argument('--medium', action='store_const', const=1,
            help='Use a large dataset.')
    parser.add_argument('--large', action='store_const', const=1,
            help='Use a large dataset.')
    arguments = parser.parse_args() # pylint: disable=invalid-name
    file_size = 'small'
    if arguments.small is None:
        arguments.small = 0
    else:
        file_size = 'small'

    if arguments.medium is None:
        arguments.medium = 0
    else:
        file_size = 'medium'

    if arguments.large is None:
        arguments.large = 0
    else:
        file_size = 'large'
    main(file_size)

