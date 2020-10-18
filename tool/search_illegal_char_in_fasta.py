fp = open(r'data\source_database_data\RNA_sequence\NPInter2_lncRNA\temp\lncRNA_sequence4.fasta', 'r')
count = 0
for line in fp.readlines():
    line = line.strip()
    if line[0]=='>':
        count = count + 1
    if count == 174 and line[0]=='>':
        print(line)
    elif count == 174:
        for char in line:
            if char != 'A' and char != 'C' and char != 'G' and char != 'T' and char != 'a' and char != 'c' and char != 'g' and char != 't':
                print(char)
                print(line)
fp.close()