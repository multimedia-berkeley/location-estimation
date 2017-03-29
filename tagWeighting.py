#tagWeighting 1.1.py runs on train_image_metadata and mediaeval2016_placing_train_photo
import ast
import sys
import itertools
import random
import collections
from math import radians, cos, sin, asin, sqrt, pow, log
from pprint import pprint 
from itertools import permutations
import time

import logging 

#NUM_LINES = 5000000
AVG_EARTH_RADIUS = 6371  # in km
#NUM_SAMPLE = 300

# sys.argv[0]: program name(tagweighting.py)
# sys.argv[1]: infile1
# sys.argv[2]: infile2
# sys.argv[3]: outfile
# sys.argv[4]: Lambda
# sys.argv[5]: w

def initialize_logger(logfilename):
    logging.basicConfig(level=logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(logfilename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    return logger

def init():
    """return the spatial aware weighting of all the tags in the dataSet.
       Write the returnVal to destination file
    :input: infile1, dataSet1 given 1
            infile2, dataSet2 given 2
            infile3, dataSet3 given(photoID, userID) 3
            outfile, destination file 4
            lambda, distance threshold, usually 40 # in km 5
            w, usually 1 6
            NUM_LINES, 7
            NUM_SAMPLE, 8
    
    Example: python tagWeighting.py train_image_metadata mediaeval2016_placing_train_photo train_image_uid.1 tagweights2.tsv 40 1 5000000 300
    :output: Returns null.
    """
    #tagDict: a dictionary of (tag, info); info is a set of (longitude, latitude)
    #tagDict: (tag, ([longitude1, latitude1], [longitude2, latitude2]))
    global logger
    logger = initialize_logger('tagweighting.log')

    start = time.time()
    tagDict = {}
    qDict = {}
    weightDict = {}
    weight = 0
    outfile = sys.argv[4]
    print(sys.argv[4])
    Lambda = float(sys.argv[5])

    w = float(sys.argv[6])

    global NUM_LINES
    global NUM_SAMPLE
    NUM_LINES = int(sys.argv[7])
    NUM_SAMPLE = int(sys.argv[8])
    
    # use parseDoc to get the tagDict : (keyID, info)
    # info is a set of (keyID, longitude, latitude)
    tagDict, numItems = parseDoc()
    logger.info("numItems")
    print(numItems)
    # get rid of the tags that are only used by one user in tagDict
    #newtagDict, newnumItems = userCheck(tagDict, numItems)
    newtagDict, newnumItems = userCheck(tagDict, numItems)
    print("tagDcit", len(tagDict))
    print("newTagDict", len(newtagDict))

    for item in newtagDict.items():
        tag = item[0]
        Nt = len(newtagDict[tag])
        if Nt > newnumItems / 20:
            #print tag, Nt
            curweightDict = {tag: 0.0001}

        else:
            qDict = find_qDict(item, Lambda)
            weight = tag_weighting(Nt, qDict, Lambda, w)
            curweightDict = {tag: (weight, Nt)}
        weightDict.update(curweightDict)

    
    #write the weight dictionary to destination file
    out = open(outfile, 'w')
    dictStr = str(weightDict)
    out.write(dictStr)
    out.close()
    l = weightDict.items()
    sorted_by_second = sorted(l, key=lambda tup: tup[1])
    pprint(sorted_by_second)
    print len(l)
    print("closed")
    print time.time() - start, 'seconds'
    return 

def parseDoc():
    """ takes in a train_image_metadata and mediaeval2016_placing_train_photo then return a tagDict: (tag, info)
        info is a set of tuples: (longitude, latitude)

        TODO: put into userID 
        :input: train_image_metadata (with pictureID and tagList)
                mediaeval2016_placing_train_photo (with pictureID and longitude, latitude)
    Example: parseDoc()
    :output: Returns: a tagDict: (tag, [(lon1, lat1), (lon2, lat2), (lon3, lat3)......])
             numItems: total number of pictures
    """
    #infile1 is normally train_image_metadata
    logger.info('parseDoc() started')

    infile1 = sys.argv[1]
    #infile2 is normally mediaeval2016_placing_train_photo
    infile2 = sys.argv[2]
    infile3 = sys.argv[3]
    inf1 = open(infile1, 'r')
    inf2 = open(infile2, 'r')
    inf3 = open(infile3, 'r')
    numItems = 0
    tagDict = collections.defaultdict(list)
    tagList = []
    tagSet = set()
    #parsing inf1 and inf2 together
    for line1, line2, line3 in itertools.izip(inf1, inf2, inf3):
        if numItems == NUM_LINES:
            #print("here1")
            break
        numItems += 1
        line_words1 = line1.split('\t')
        #print(line_words1)
        # pictureID
        pictureID = line_words1[0]
        # adding into tagSet: (tag, pictureID)
        tagString = line_words1[3]
        #print(pictureID)
        #print(tagString)
        
        # eval string representation of tagList to list
        #tagList = eval(tagString)
        tagList = tagString.split(', ')
        line_words2 = line2.split('\t')

        # longitude
        longitude = line_words2[2]
        # latitude
        latitude = line_words2[3]

        line_words3 = line3.split('\t')
        userID = line_words3[1]
        for tag in tagList:
            tag = tag.replace(" ", "")
            tagSet.add((tag, (userID, longitude, latitude)))
   
    
    for k, v in tagSet:
        tagDict[k].append(v)
        # here the tagDict consists of (tag, a set of info)
        # info: (userID, longitude, latitude)
    #print(len(tagDict))
    #sys.exit(1)
    logger.info('parseDoc() end')
    return tagDict, numItems

#not in use now
def userCheck(tagDict, numItems):
    """ get rid of the tags that are only used by one user.
    :input: tagDict 
                   key: tag
                   value: (userID, longitude, latitude)
            numItems: original number of items in tagDict

    Example: userCheck(tagDict)

    :output: a new Dictionary newtagDict
             newnumItems: a new number of items
    """
    userList = set()
    newtagDict = {}
    newnumItems = 0
    for key in tagDict.keys():
        valueList = tagDict[key]
        for value in valueList:
            userList.add(value[0])    
            if len(userList) > 1: # tag used by more than one user 
                newtagDict[key] = valueList
                newnumItems += 1 
        userList = set()
    return newtagDict, newnumItems


# not in use now
def checkEqual(iterator):
    """ check if the iterator is composed of same element
    :input: an iterator

    Example: checkEqual(iterator)

    :output: True or False
    """
    if len(iterator) == 1:
        return False
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(first == rest for rest in iterator)
    except StopIteration:
        return True


def haversine(point1, point2, miles=False):
    """ Calculate the great-circle distance bewteen two points on the Earth surface.
    :input: two 2-tuples, containing the latitude and longitude of each point
    in decimal degrees.
    Example: haversine((45.7597, 4.8422), (48.8567, 2.3508))
    :output: Returns the distance bewteen the two points.
    The default unit is kilometers. Miles can be returned
    if the ``miles`` parameter is set to True.
    """
    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * asin(sqrt(d))

    if miles:
        return h * 0.621371  # in miles
    else:
        return h  # in kilometers


def tag_weighting(Nt, q_Dict, Lambda, w):
        """ Calculate the spatial-aware Tag Weighting for the tag associated with q_Dict(a dictionary of all of the pictures associated with the tag)
        :input:  Nt: total occurances of tag from q_Dict
                 q_Dict: a dictionary of tag that needs to be weighted, see more at find_qDict()
                 Lambda, distance threshold, normally 40 # in km
                 w, normally 1
        Example: tag_weighting(q_Dict 40, 1)
        
        :output: Returns the weight of tag associated with q_Dict.
        """
        #print Nt, q_Dict, Lambda, w
        returnVal = 0
        qVal = 0
        q_Len = 0
        # q_List is a list of ((lon1, lat1), (lon2, lat2)) pairs and 0.0 (if not satisfy lambda)
        q_List = q_Dict.values()[0]
        q_newList = []
        # get rid of items that does not satisfy lambda in q_List and put the rest into q_newList     
        #print q_List
        for item in q_List:
            if item != 0.0:
                q_newList.append(item)
        # if tag only appear once, no need to go through the rest
        if len(q_newList) == 1:
            qVal += pow(1, w)
        
        
        else:
            
        # sort q_newList into a dictionary, thereby we have (p, [q1, q2, q3, .....])
            pair_Dict = collections.defaultdict(list)
            for k,v in q_newList:
                pair_Dict[k].append(v)
        # get a list of keys from pair_Dict: thereby we have [p1, p2, p3, p4 ......]
            key_List = pair_Dict.keys()
        # for each key in key_List, get a list of its values, and compute tag weighting on this key
            for key in key_List:
                q_Len = 0
                value_List = pair_Dict[key]
                for value in value_List:
                    if value != 0.0:
                        q_Len += 1
                qVal += pow(q_Len, w)
        #print qVal, Nt
        returnVal = float(qVal) / pow(Nt, 2)
        returnVal = returnVal * log(Nt) 
        return returnVal    



def find_qDict(item, Lambda):
        """ Find a dictionary of values within distance threshold from keys
        :input: item: an item of tagDict
                tagDict  = (tag, [(userID1, lon1, lat1), (userID2, lon2, lat2), (userID3, lon3, lat3)......])
                Lambda: distance threshold
        Example: find_qDict(item, 40)
        :output: q_Dict: ((givenLon, givenLat), [(lon1, lat1), (lon2, lat2), ......])
        """

        s = []
        locationList = []
        newList = []
        q_Dict = collections.defaultdict(list) 
        if len(item[1]) == 1:
            q_Dict = {item[0]: [(item[1][0][0], item[1][0][1], item[1][0][2])]}
            return q_Dict
        i = 0
        tag = item[0]
        l_lonlat = item[1]

        for lonlat in l_lonlat[:NUM_SAMPLE]:
            userID = lonlat[0]
            Lon = float(lonlat[1])
            Lat = float(lonlat[2])
            locationList.append((userID, Lon, Lat))
        """   
        while i <= len(item[1]) - 1:
            if i >= NUM_SAMPLE:
                break
            Lon = float(item[1][i][0])
            Lat = float(item[1][i][1])
            locationList.append((Lon, Lat))
            i += 1
        #while i <= len(item[1]) - 1:
        ##3   Lat = float(item[1][i][1])
         #   i += 1
        """
        newList = itertools.permutations(locationList, 2)
        for item in newList:
            givenUserID = item[0][0]
            givenLon = item[0][1]
            givenLat = item[0][2]
            nextUserID = item[1][0]
            nextLon = item[1][1]
            nextLat = item[1][2]
            distance = haversine((givenLat, givenLon), (nextLat, nextLon))
            if distance <= Lambda:
                s.append((tag,((givenUserID, givenLon, givenLat), (nextUserID, nextLon, nextLat))))
            else:
                s.append((tag, 0.0))
        for k, v in s:
            q_Dict[k].append(v)
        return q_Dict

if __name__ == '__main__':
    init()  



