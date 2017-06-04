import argparse
import subprocess

def main(command, all_num_gpus, all_batch_sizes, output_folder):
  subprocess.call("rm -rf %s; mkdir -p %s" % (output_folder, output_folder),
                  shell=True)
  for batch_size in all_batch_sizes:
    subprocess.call("mkdir -p %s/batch_size=%d" % (output_folder, batch_size),
                    shell=True)
    for num_gpus in all_num_gpus:
      subprocess.call("echo \"%s --batch_size=%d --num_gpus=%d\n\" >> %s/batch_size=%d/gpus=%d.out" %
                      (command, batch_size, num_gpus, output_folder, batch_size, num_gpus),
                      shell=True)
      subprocess.call("%s --batch_size=%d --num_gpus=%d >> %s/batch_size=%d/gpus=%d.out 2>/dev/null" %
                      (command, batch_size, num_gpus, output_folder, batch_size, num_gpus),
                      shell=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description=("Sweep num_gpus and batch_size parameters")
  )
  parser.add_argument('-c', "--command", type=str, required=True,
                      help="Command to run")
  parser.add_argument('-g', "--num_gpus", nargs='+', type=int,
                      help="List of number of GPUs to sweep through")
  parser.add_argument('-b', "--batch_size", nargs='+', type=int,
                      help="List of batch sizes to sweep through")
  parser.add_argument('-o', "--output_folder", type=str, required=True,
                      help="Output folder to dump logs")

  cmdline_args = parser.parse_args()
  opt_dict = vars(cmdline_args)

  main(opt_dict["command"], opt_dict["num_gpus"], opt_dict["batch_size"],
       opt_dict["output_folder"])
