# Parse the performance testing output measurements to get results in csv format

import csv
import glob
import os
from os.path import basename
import pprint
import sys

result_dir = ''

input_file_names = []

"""
Parsed data from the output files
1-based index
List (runs) of lists (files) of dictionaries (metrics)- 
    3+1 rows ('', out1, out2, out3) and columns (output files) for each input (100, 1000, 1500 etc.)
"""
out = [[{}]] 
perf = [[{}]]
cuda = [[{}]]

"""
Returns the time value from a line in a format like
"Wall Time =  7.4067 seconds"
"""
def extract_time(time_line):
    time_str = time_line.split(" = ")[-1]
    time = time_str.split(" ")[0]
    return time

"""
Extracts the following from the ".out" files:
  - wall_time
  - cpu_time 
"""
def process_out_files(out_files):
    for run in [1, 2, 3]:
        out.append([])

        for file in out_files[run]:
            with open(file) as f:
                data = f.readlines()

            [wall_time_line, cpu_time_line] = data[-2:]
            wall_time = extract_time(wall_time_line)
            cpu_time = extract_time(cpu_time_line)

            out[run].append({"wall_time": wall_time, "cpu_time": cpu_time})

"""
Extracts the following from the ".perf" files:
  - instructions
  - cycles
  - context-switches
  - cache-references 
  - cache-misses
"""
def process_perf_files(perf_files):
    for run in [1, 2, 3]:
        perf.append([])

        for file in perf_files[run]:
            with open(file) as f:
                data = f.readlines()

            perf_dict = {}
            for metric_line in data:
                entries = metric_line.split(",")
                value = entries[0]
                metric = entries[2]

                perf_dict[metric] = value

            perf[run].append(perf_dict)            

"""
Extracts the profiling data from the ".cuda" files:
"""
def process_cuda_files(cuda_files):
    # TODO:
    row = ["2", "Marie", "California"]
    """
    with open('people.csv', 'r') as readFile:
        reader = csv.reader(readFile)
        lines = list(reader)
        lines[2] = row

    for run in [1, 2, 3]:
        cuda.append([])

        for file in perf_files[run]:
            with open(file, mode='r') as f:
                csv_reader = csv.DictReader(f)
                line_count = 0

                for row in csv_reader:
                    if line_count == 0:
                        print(f'Column names are {", ".join(row)}')
                        line_count += 1

                    print(f'\t{row["name"]} works in the {row["department"]} department, and was born in {row["birthday month"]}.')
                    line_count += 1

            # cuda[run].append(cuda_dict)  
    """

def get_input_file_names():
    files = [file for file in glob.glob(os.path.join(result_dir,"*.out1"))]
    files = [basename(file).split(".")[0] + ".in" for file in files]

    return files

def get_files(extension):
    files_1 = [file for file in glob.glob(os.path.join(result_dir,"*." + extension + "1"))]
    files_2 = [file for file in glob.glob(os.path.join(result_dir,"*." + extension + "2"))]
    files_3 = [file for file in glob.glob(os.path.join(result_dir,"*." + extension + "3"))]

    return [[], files_1, files_2, files_3] # 1-based index

"""
Returns the run that needed the lowest wall clock time.
"""
def get_best_run(input_idx):
    time_1 = out[1][input_idx].get("wall_time")
    time_2 = out[2][input_idx].get("wall_time")
    time_3 = out[3][input_idx].get("wall_time")

    if (time_1 < time_2 and time_1 < time_3):
        return 1
    elif (time_2 < time_1 and time_2 < time_3):
        return 2
    else:
        return 3

def write_results_csv():
    results_csv_file = os.path.join(result_dir, 'results.csv')

    with open(results_csv_file, mode='w', newline='') as results_csv:
        writer = csv.writer(results_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        csv_header = [
            'Input File',
            'Wall Time',
            'CPU Time',
        ]
        
        if HAS_PERF:
            csv_header.extend([
                'instructions',
                'cycles',
                'context-switches',
                'cache-references',
                'cache-misses',
                'IPC',
            ])

        if HAS_CUDA:
            # TODO: cuda
            csv_header.extend([
                'c',
            ])

        writer.writerow(csv_header)

        for idx in range(len(input_file_names)):
            row = []
            # row.append(input_file_names[idx]) 
            # Only use the number as it will be easier to generate plots
            row.append(input_file_names[idx].split(".")[0])

            run = get_best_run(idx) # Only output best of Run 1, Run 2 and Run 3

            row.append(out[run][idx].get("wall_time"))
            row.append(out[run][idx].get("cpu_time"))

            if HAS_PERF:
                row.append(perf[run][idx].get("instructions"))
                row.append(perf[run][idx].get("cycles"))
                row.append(perf[run][idx].get("context-switches"))
                row.append(perf[run][idx].get("cache-references"))
                row.append(perf[run][idx].get("cache-misses"))
                row.append(float(perf[run][idx].get("instructions")) / float(perf[run][idx].get("cycles")))

            if HAS_CUDA:  
                # TODO: cuda
                row.append(cuda[run][idx].get("c"))

            writer.writerow(row)

if __name__ == "__main__":
    result_dir = sys.argv[1]

    input_file_names = get_input_file_names()

    out_files = get_files("out")
    perf_files = get_files("perf")
    cuda_files = get_files("cuda")

    HAS_PERF = False
    HAS_CUDA = False

    if len(perf_files[1]) != 0:
        HAS_PERF = True
    if len(cuda_files[1]) != 0:
        HAS_CUDA = True

    process_out_files(out_files)
    if HAS_PERF:
        process_perf_files(perf_files)
    if HAS_CUDA:
        process_cuda_files(cuda_files)

    write_results_csv()
