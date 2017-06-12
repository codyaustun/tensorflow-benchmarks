import argparse
import subprocess
import multiprocessing

lock = multiprocessing.Lock()

def spawn_process(command, server, docker_image, filename, return_output):
  # TODO: Don't start a new docker container here; have a setup phase where docker containers
  # are started; then have a teardown phase where docker containers are killed.
  docker_cmd = """nvidia-docker run -v /mnt:/mnt --net=host %s /bin/bash -c 'cd ~/tensorflow_benchmarks/scripts/tf_cnn_benchmarks; %s 2>/dev/null'""" % (docker_image, command)
  ssh_cmd = "ssh -n %s -o StrictHostKeyChecking=no \"%s\"" % (server, docker_cmd)
  if return_output:
    output = subprocess.check_output(ssh_cmd, shell=True)
    with lock:
        with open(filename, 'a') as f:
            f.write(output + "\n")
  else:
    subprocess.check_output(ssh_cmd, shell=True)

def main(command, servers, num_gpus_per_node, all_num_nodes, all_batch_sizes,
         output_folder, docker_image, input_path, output_path, time_interval):
  subprocess.call("rm -rf %s; mkdir -p %s" % (output_folder, output_folder),
                  shell=True)
  throughputs = dict()

  for batch_size in all_batch_sizes:
    subprocess.call("mkdir -p %s/batch_size=%d" % (output_folder, batch_size),
                    shell=True)
    for num_nodes in all_num_nodes:
      # Log command to output file.
      filename = "%s/batch_size=%d/gpus=%d.out" % (output_folder, batch_size, num_nodes)
      with open(filename, 'w') as f:
        command_to_execute = command
        if input_path is not None:
          command_to_execute += " --train_dir=%s" % input_path
        worker_hosts = ",".join(["%s:50000" % servers[i] for i in xrange(num_nodes)])
        ps_hosts = ",".join(["%s:50001" % servers[i] for i in xrange(num_nodes)])
        command_to_execute += " --worker_hosts=%s --ps_hosts=%s --num_gpus=%d --batch_size=%d" % (
          worker_hosts, ps_hosts, num_gpus_per_node, batch_size)
        worker_commands = ["%s --job_name=worker --task_index=%d" % (command_to_execute, i) for i in xrange(num_nodes)]
        ps_commands = ["CUDA_VISIBLE_DEVICES='' %s --job_name=ps --task_index=%d" % (
          command_to_execute, i) for i in xrange(num_nodes)]
        f.write("%s\n\n" % ("\n".join(worker_commands)))

      # Start PS threads.
      ps_threads = []
      for i in xrange(num_nodes):
        ps_thread = multiprocessing.Process(
          target=spawn_process,
          args=(ps_commands[i], servers[i], docker_image, filename, False))
        ps_threads.append(ps_thread)
        ps_thread.start()

      # Start worker threads.
      worker_threads = []
      for i in xrange(num_nodes):
        worker_thread = multiprocessing.Process(
          target=spawn_process,
          args=(worker_commands[i], servers[i], docker_image, filename, True))
        worker_threads.append(worker_thread)
        worker_thread.start()

      checkpoint_thread = None
      if input_path is not None and output_path is not None:
        checkpoint_command = "python backup_checkpoints.py -i %s -o %s -t %d" % (
          input_path, output_path, time_interval)
        checkpoint_thread = multiprocessing.Process(
          target=spawn_process,
          args=(checkpoint_command, servers[0], docker_image, filename, False))
        checkpoint_thread.start()

      for i in xrange(num_nodes):
        worker_threads[i].join()

      for i in xrange(num_nodes):
        ps_threads[i].terminate()

      if checkpoint_thread is not None:
        checkpoint_thread.terminate()

      # Kill the running docker container with the parameter server instance.
      for i in xrange(num_nodes):
        cmd = "nvidia-docker kill $(nvidia-docker ps -q)"
        subprocess.check_output("ssh -n %s -o StrictHostKeyChecking=no '%s'" % (servers[i],
                                                                                cmd),
                                shell=True)

      with open(filename, 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
          if "total images/sec:" in line:
            throughput = float(line.split(": ")[1].strip())
            if (batch_size, num_nodes) in throughputs:
              if throughput > throughputs[(batch_size, num_nodes)]:
                throughputs[(batch_size, num_nodes)] = throughput
            else:
              throughputs[(batch_size, num_nodes)] = throughput

  # Log summary to output.
  print "\t".join([""] + ["# GPUs = %d" % num_nodes * num_gpus_per_node
                          for num_nodes in all_num_nodes])
  for batch_size in all_batch_sizes:
    values = ["Batch size = %d" % batch_size]
    for num_nodes in all_num_nodes:
      values.append(str(throughputs[(batch_size, num_nodes)]))
    print "\t".join(values)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description=("Sweep num_gpus and batch_size parameters")
  )
  parser.add_argument('-s', "--servers", nargs='+', type=str,
                      help="Servers to run commands on")
  parser.add_argument('-c', "--command", type=str, required=True,
                      help="Command to run")
  parser.add_argument('-g', "--num_gpus_per_node", type=int, required=True,
                      help="Number of GPUs to use per node")
  parser.add_argument('-n', "--num_nodes", nargs='+', type=int,
                      help="List of number of nodes to sweep through")
  parser.add_argument('-b', "--batch_size", nargs='+', type=int,
                      help="List of batch sizes to sweep through")
  parser.add_argument('-o', "--output_folder", type=str, required=True,
                      help="Output folder to dump logs")
  parser.add_argument('-d', "--docker_image", type=str, required=True,
                      help="Name of docker image")
  parser.add_argument('-i', "--input_path", type=str, default=None,
                      help="Training directory where checkpoints are dumped")
  parser.add_argument('-p', "--output_path", type=str, default=None,
                      help="Path to dump checkpointed models at regular intervals of time")
  parser.add_argument('-t', "--time_interval", type=int, default=900,
                      help="Time interval (in seconds) between model checkpoint dumps")

  cmdline_args = parser.parse_args()
  opt_dict = vars(cmdline_args)

  main(opt_dict["command"], opt_dict["servers"], opt_dict["num_gpus_per_node"],
       opt_dict["num_nodes"], opt_dict["batch_size"], opt_dict["output_folder"],
       opt_dict["docker_image"], opt_dict["input_path"], opt_dict["output_path"],
       opt_dict["time_interval"])
