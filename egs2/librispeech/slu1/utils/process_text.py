fp1 = open("text.1", 'w')
with open("text") as fp:
    for i, line in enumerate(fp, 1):
        if i % 10000 == 0:
            print(i)
        words, labels = line.strip().split('\t')
        words = words.split()[:-1]
        uttids, words = words[0], words[1:]
        labels = labels.split()[:-1]
        assert len(words) == len(labels)
        new_words = []
        for i in range(1, len(words)):
            if labels[i] == labels[i-1]:
                new_words.append(words[i])
            else:
                if labels[i] == "O":
                    new_words.append(f"E-{labels[i-1]}")
                    new_words.append(words[i])
                else:
                    if labels[i-1] == "O":
                        new_words.append(f"B-{labels[i]}")
                        new_words.append(words[i])
                    else:
                        new_words.append(f"E-{labels[i-1]}")
                        new_words.append(f"B-{labels[i]}")
                        new_words.append(words[i])
        if labels[-1] != "0":
            new_words.append(f"E-{labels[-1]}")
        fp1.write(f"{uttid} ")
        fp1.write(" ".join(new_words))
        fp1.write("\n")
