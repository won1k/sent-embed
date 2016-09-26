import sys
import argparse
import re
import h5py
import numpy as np

class Indexer:
    def __init__(self):
        self.counter = 1
        self.d = {}
        self.rev = {}
        self.chunk_d = {}
        self.pos_d = {}
        self._lock = False
        self.max_len = 0

    def convert(self, w):
        if w not in self.d:
            if self._lock:
                return self.d["<unk>"]
            self.d[w] = self.counter
            self.rev[self.counter] = w
            self.counter += 1
        return self.d[w]

    def lock(self):
        self._lock = True

    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.iteritems()]
        items.sort()
        for v, k in items:
            print >>out, k, v
        out.close()

def get_data(args):
    target_indexer = Indexer()
    #add special words to indices in the target_indexer
    target_indexer.convert("</s>")

    def text_output(datafile, sentences):
        out = open(datafile.split(".")[0] + "_ordered.txt", "w")
        idx_to_word = dict([(v, k) for k, v in target_indexer.d.iteritems()])
        for length, sent_list in sentences.iteritems():
            for sentence in sent_list:
                print >>out, ' '.join([idx_to_word[word] for word in sentence])
        out.close()

    def convert(data_x, data_y, outfile):
        # Parse and convert data
        with open(data_x, 'r') as f:
            x = f.readlines()
        with open(data_y, 'r') as g:
            y = g.readlines()
        sentences = {}
        outputs = {}
        for i, orig_sentence in enumerate(x):
            orig_sentence = orig_sentence.strip().split() + ["</s>"]
            sentence = [target_indexer.convert(w) for w in orig_sentence]
            output_sentence = y[i].strip().split() + ["</s>"]
            output = [target_indexer.convert(w) for w in output_sentence]
            length = len(sentence)
            target_indexer.max_len = max(target_indexer.max_len, length)
            if length in sentences:
                sentences[length].append(sentence)
                outputs[length].append(output)
            else:
                sentences[length] = [sentence]
                outputs[length] = [output]
        
        sent_lens = sentences.keys()

        # Reoutput raw data ordered by length
        text_output(data_x, sentences)

        # Output HDF5 for torch
        f = h5py.File(outfile, "w")
        f["sent_lens"] = np.array(sent_lens, dtype=int)
        for length in sent_lens:
            f[str(length)] = np.array(sentences[length], dtype=int)
            f[str(length) + "output"] = np.array(outputs[length], dtype = int)
        f["max_len"] = np.array([target_indexer.max_len], dtype=int)
        f["nfeatures"] = np.array([target_indexer.counter - 1], dtype=int)
        f["nclasses"] = np.array([target_indexer.counter - 1], dtype=int)

    convert(args.train_x, args.train_y, args.outputfile + ".hdf5")
    target_indexer.lock()
    convert(args.test_x, args.test_y, args.outputfile + "_test" + ".hdf5")
    target_indexer.write(args.outputfile + ".dict")

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('train_x', help="Raw chunking text file", type=str) # data/prefix.txt
    parser.add_argument('train_y', help="Raw chunking text file", type=str) # data/infix.txt
    parser.add_argument('test_x', help="Raw chunking test text file", type=str) # data/prefix_test.txt
    parser.add_argument('test_y', help="Raw chunking test text file", type=str) # data/infix_test.txt
    parser.add_argument('outputfile', help="HDF5 output file", type=str) # data/trees
    args = parser.parse_args(arguments)

    # Do conversion
    get_data(args)

if __name__ == '__main__':
   sys.exit(main(sys.argv[1:]))
