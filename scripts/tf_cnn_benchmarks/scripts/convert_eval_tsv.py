import argparse
import os
import subprocess
import sys

def main(input_path, output_path, batch_size, num_images):
  input_lines = open(input_path, 'r').read().strip()
  input_lines = input_lines.split("\n")

  with open(output_path, 'w') as f:
    line_id = 0
    for input_line in input_lines:
      input_line_tokens = input_line.split("\t")
      if line_id == 0:
        input_line_tokens[1] = "Epoch"
      else:
        step = float(input_line_tokens[1])
        epoch = int(round((step * batch_size) / float(num_images)))
        input_line_tokens[1] = str(epoch)
      f.write("\t".join(input_line_tokens) + "\n")
      line_id += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description=("Convert old eval tsv to new eval tsv")
  )
  parser.add_argument('-i', "--input_path", type=str, required=True,
                      help="Input path")
  parser.add_argument('-o', "--output_path", type=str, required=True,
                      help="Output path")
  parser.add_argument('-b', "--batch_size", type=int, required=True,
                      help="Batch size")
  parser.add_argument('-n', "--num_images", type=int, default=1281167,
                      help="Number of images")

  cmdline_args = parser.parse_args()
  opt_dict = vars(cmdline_args)

  main(opt_dict["input_path"], opt_dict["output_path"], opt_dict["batch_size"],
       opt_dict["num_images"])
