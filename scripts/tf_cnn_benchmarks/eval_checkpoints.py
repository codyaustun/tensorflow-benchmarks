import argparse
import os
import subprocess
import sys
import time

def main(input_path, command):
  cnt = 0
  with open(os.path.join(input_path, "README.md"), 'r') as f:
    output = f.read()
    output_lines = output.split('\n')
    time_interval = int(output_lines[0].split(" = ")[1].strip().rstrip(" secs"))
    original_path = output_lines[1].split(" = ")[1].strip()
  print "Time (in secs)\tNumber of minibatches\tTop 1 accuracy\tTop 5 accuracy"
  while True:
    ckpt_path = ("%5d" % cnt).replace(' ', '0')
    full_ckpt_path = os.path.join(input_path, ckpt_path)
    if not os.path.exists(full_ckpt_path):
      break
    if len(os.listdir(full_ckpt_path)) <= 2:
      cnt += 1
      continue
    subprocess.check_output("mkdir -p %s; rm %s/*; cp %s/* %s" %
                            (original_path, original_path, full_ckpt_path, original_path),
                            shell=True)
    full_command = command + " --train_dir=%s 2>/dev/null" % original_path
    output = subprocess.check_output(full_command, shell=True)
    for line in output.split('\n'):
      if "Precision" in line and "Recall" in line:
        tokens = line.split(", ")  # TODO: Nasty hack, make more robust.
        precision_at_1 = float(tokens[0].split()[-1])
        recall_at_5 = float(tokens[1].split()[-1])
        global_step = int(tokens[2].split()[3])
        stats = [(cnt + 1) * time_interval, global_step, precision_at_1, recall_at_5]
        print "\t".join([str(stat) for stat in stats])
        sys.stdout.flush()
    cnt += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description=("Backup model checkpoints periodically")
  )
  parser.add_argument('-i', "--input_path", type=str, required=True,
                      help="Path to dumped model checkpoints")
  parser.add_argument('-c', "--command", type=str, required=True,
                      help="Command to evaluate each individual checkpoint")

  cmdline_args = parser.parse_args()
  opt_dict = vars(cmdline_args)

  main(opt_dict["input_path"], opt_dict["command"])
