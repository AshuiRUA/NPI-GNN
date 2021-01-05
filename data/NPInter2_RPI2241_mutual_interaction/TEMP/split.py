f = open(r'interactionKey_RPI2241_mutual', 'r')
count = 0
for line in f.readlines():
    f_temp = open(f'./interactionKey_RPI2241_mutual_{count}', 'w')
    f_temp.write(line)
    f_temp.close()
    count += 1
