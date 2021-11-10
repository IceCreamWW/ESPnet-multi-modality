import argparse
import textgrid
import logging
import glob
import os

def process_textgrid(filepath):
    tg = textgrid.TextGrid.fromFile(filepath)
    words, word_durs, phns, phn_durs = [], [], [], []
    # get word alignment
    for item in tg[0]:
        words.append(item.mark if item.mark else "<sil>")
        word_durs.append(round((item.maxTime - item.minTime) * 100))

    # get phn alignment
    for item in tg[1]:
        phns.append(item.mark)
        phn_durs.append(round((item.maxTime - item.minTime) * 100))
    if not phns[-1]:
        phns = phns[:-1]
        phn_durs = phn_durs[:-1]

    return (phns, phn_durs), (words, word_durs)

def process_dir(dirpath):
    phns_fp = open(os.path.join(dirpath, "phns"), 'w')
    phn_durs_fp = open(os.path.join(dirpath, "phn_durs"), 'w')
    words_fp = open(os.path.join(dirpath, "words"), 'w')
    word_durs_fp = open(os.path.join(dirpath, "word_durs"), 'w')

    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        if not filename.endswith("TextGrid"):
            logging.debug(f"{filepath} is not a textgrid file, ignore it")
            continue
        uttid = os.path.splitext(filename)[0]
        (phns, phn_durs), (words, word_durs) = process_textgrid(filepath)

        phns_fp.write(f"{uttid} {' '.join(phns)}\n")
        phn_durs_fp.write(f"{uttid} {' '.join(['%d' % dur for dur in phn_durs])}\n")
        # phn_durs_fp.write(f"{uttid} {' '.join(['%.2f' % dur for dur in phn_durs])}\n")
        words_fp.write(f"{uttid} {' '.join(words)}\n")
        word_durs_fp.write(f"{uttid} {' '.join(['%d' % dur for dur in word_durs])}\n")
        # word_durs_fp.write(f"{uttid} {' '.join(['%.2f' % dur for dur in word_durs])}\n")

    phns_fp.close()
    phn_durs_fp.close()
    words_fp.close()
    word_durs_fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process text grid alignment')
    parser.add_argument('--dir', type=str, help='target dir to be processed')
    args = parser.parse_args()
    logging.basicConfig(encoding='utf-8', level=logging.INFO)

    dirs = glob.glob(f"{args.dir}/*/*")
    for idir, dirpath in enumerate(dirs, 1):
        logging.info(f"{idir}/{len(dirs)}")
        process_dir(dirpath)


