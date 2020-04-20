import time
import logging
import argparse
from pathlib import Path
import numpy as np
from word_vectors.read import read_glove, read_w2v, read_dense
from word_vectors.write import write_glove, write_dense

LOC = Path(__file__).parent.resolve() / "data"
W2V = LOC / "GoogleNews-vectors-negative300.bin"
GLOVE = LOC / "GoogleNews.glove"
DENSE = LOC / "GoogleNews.dense"

logging.basicConfig(format="[Benchmark] %(message)s", level="INFO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", "-t", default=5, type=int)
    args = parser.parse_args()
    trials = args.trials
    vsz = None
    dim = None
    if not W2V.exists():
        logging.info("%s not found. Please run `get_data.sh`" % W2V.name)
    if not GLOVE.exists() or not DENSE.exists():
        w, wv, len_ = read_w2v(W2V, stats=True)
        if not GLOVE.exists():
            logging.info("%s not found, Creating..." % GLOVE.name)
            write_glove(w, wv, GLOVE)
        if not DENSE.exists():
            logging.info("%s not found, Creating..." % DENSE.name)
            write_dense(w, wv, len_, DENSE)
    logging.info("Beginning Benchmark...")
    if vsz is None or dim is None:
        w, wv = read_dense(DENSE)
    logging.info("Vocab Size: %d" % len(w))
    logging.info("Vector dim: %d" % wv.shape[1])
    del w
    del wv
    if trials == 1:
        logging.info("Running %d trial." % trials)
    else:
        logging.info("Running %d trials." % trials)
    logging.info("Glove size: %.2fGB" % (GLOVE.stat().st_size / (1024 ** 3)))
    glove = benchmark(GLOVE, read_glove, trials)
    logging.info("Glove: %.2f \u00B1 %.2f" % (np.mean(glove), np.std(glove)))
    logging.info("W2V size: %.2fGB" % (W2V.stat().st_size / (1024 ** 3)))
    w2v = benchmark(W2V, read_w2v, trials)
    logging.info("W2V: %.2f \u00B1 %.2f" % (np.mean(w2v), np.std(w2v)))
    logging.info("Dense size: %.2fGB" % (DENSE.stat().st_size / (1024 ** 3)))
    dense = benchmark(DENSE, read_dense, trials)
    logging.info("Dense: %.2f \u00B1 %.2f" % (np.mean(dense), np.std(dense)))


def benchmark(file_name, read, trials):
    times = []
    for _ in range(trials):
        t0 = time.time()
        _ = read(file_name)
        times.append(time.time() - t0)
    return times


if __name__ == "__main__":
    main()
