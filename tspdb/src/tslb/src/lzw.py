######################################################
#
# LZW compression
#
######################################################
from io import StringIO

def get_string(uncomp_numbers, n):
    # uncomp_number is a list of numbers
    # max(n) = 256 is the alphabet size
    # make dictionary
    dictionary = {i : chr(i) for i in range(n)}
    # convert number list to string
    uncompressed = str()        
    for i in uncomp_numbers:
        uncompressed = uncompressed + dictionary[i]
    return uncompressed

def compress(uncompressed):
    # uncompressed is a string
    """Compress a string to a list of output symbols."""
    # Build the dictionary.
    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}
    w = ""
    result = []
    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            # Add wc to the dictionary.
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
    # Output the code for w.
    if w:
        result.append(dictionary[w])
    return result

def decompress(compressed):
    # compressed is a list of codewords

    """Decompress a list of output ks to a string."""
    dict_size = 256
    dictionary = dict((i, chr(i)) for i in range(dict_size))
 
    # use StringIO, otherwise this becomes O(N^2)
    # due to string concatenation in a loop
    result = StringIO()
    w = chr(compressed.pop(0))
    result.write(w)
    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError('Bad compressed k: %s' % k)
        result.write(entry)
 
        # Add w+entry[0] to the dictionary.
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
 
        w = entry
    return result.getvalue()

def lzw_compression_ratio(uncomp_numbers, n):
    uncompressed = get_string(uncomp_numbers, n)
    # compression
    compressed = compress(uncompressed)
    compression_ratio = len(compressed)/len(uncompressed)

    # print("uncomp_numbers   : ", uncomp_numbers)
    # print("uncompressed     : ", uncompressed)
    # print("compressed       : ", compressed)
    # print("compression ratio: ",compression_ratio)

    return compression_ratio

