import os, glob

if __name__ == '__main__':
    txt_list = glob.glob("/media/pisces/曙光/smoke/train_txt/*.txt")
    for txt_item in txt_list:
        with open(txt_item) as f:
            lines = f.readlines()
        with open(txt_item, 'w') as f:
            for line in lines:
                line_split = line.strip().split()
                line_split[0] = '1'
                for i in range(1, 5):
                    if len(line_split[i]) > 8:
                        line_split[i] = line_split[i][0:8]
                    elif len(line_split[i]) < 8:
                        line_split[i] = line_split[i].ljust(8, '0')
                f.write(
                    line_split[0] + ' ' +
                    line_split[1] + " " +
                    line_split[2] + " " +
                    line_split[3] + " " +
                    line_split[4]+'\n')
        pass

