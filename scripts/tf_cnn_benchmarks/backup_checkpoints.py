import argparse
import os
import subprocess
import time

def main(input_path, output_path, time_interval):
  cnt = 0
  subprocess.call("rm -rf %s; mkdir -p %s" % (output_path, output_path), shell=True)
  subprocess.call("echo \"Time interval = %d secs\" > %s" % (time_interval,
                  os.path.join(output_path, "README.md")), shell=True)
  subprocess.call("echo \"Original checkpoint path = %s\" >> %s" % (input_path,
                  os.path.join(output_path, "README.md")), shell=True)
  while True:
    time.sleep(time_interval)
    ckpt_path = ("%5d" % cnt).replace(' ', '0')
    full_output_path = os.path.join(output_path, ckpt_path)
    print "Backing up checkpoint to %s..." % full_output_path
    subprocess.call("mkdir -p %s" % full_output_path, shell=True)
    subprocess.call("cp %s/checkpoint %s/" % (input_path, full_output_path), shell=True)
    subprocess.call("cp %s/graph.pbtxt %s/" % (input_path, full_output_path), shell=True)
    subprocess.call("mv %s/model* %s/" % (input_path, full_output_path), shell=True)
    cnt += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description=("Backup model checkpoints periodically")
  )
  parser.add_argument('-i', "--input_path", type=str, required=True,
                      help="Path of checkpoint to backup")
  parser.add_argument('-o', "--output_path", type=str, required=True,
                      help="Path to dump model checkpoints")
  parser.add_argument('-t', "--time_interval", type=int, required=True,
                      help="Time interval (in seconds) between model checkpoint dumps")

  cmdline_args = parser.parse_args()
  opt_dict = vars(cmdline_args)

  main(opt_dict["input_path"], opt_dict["output_path"], opt_dict["time_interval"])
