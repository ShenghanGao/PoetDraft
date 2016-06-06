# coding: utf8
import operator
import string

input_file = ['chars_tang.txt', 'chars_song.txt']
output_file = 'chars_tatol.txt'


def main():
  counts = dict()
  for i in range(2):
    with open(input_file[i],'r') as in_f:
      for line in in_f:
        entry = line.split(':')
        c = entry[0].decode('utf8')
        if entry[1].strip():
          counts[c] = counts.get(c, 0) + int(entry[1].strip())
    in_f.close()

  sorted_counts = sorted(counts.items(), key=operator.itemgetter(1))
  sorted_counts = reversed(sorted_counts)
  j = 0
  with open(output_file, 'w+') as out_f:
    for i in sorted_counts:
      out_f.write("[" + str(j) + "]"+ i[0].encode('utf8') + ":" + str(i[1]) + ' ')
      #out_f.write("[" + str(j) + "] "+ i[0].encode('utf8') + ' ')
      if(j % 10 == 9):
        out_f.write("\n")
      j = j + 1
  out_f.close()

  
if __name__ == '__main__':
  main()
