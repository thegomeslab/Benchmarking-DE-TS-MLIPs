import os
import sys


def process_tsoutfile(qcout_file: str) -> None:
    directory = os.path.dirname(qcout_file)
    with open(qcout_file,'r') as f:
        data = f.readlines()
        for i,line in enumerate(data):
            if '**  OPTIMIZATION CONVERGED  **' in line:
                opt_conv = i
                cont = True
            if 'Z-matrix Print:' in line:
                z_mat = i
            if 'Final energy is' in line:
                energy = line.split()[-1]
            if " **                       VIBRATIONAL ANALYSIS                       **" in line:
                freq_line = i+11
        num_atoms = z_mat-opt_conv-6
        final_structure = data[opt_conv+5:z_mat-1]
    freq = float(data[freq_line].split()[1])
    outfile = qcout_file.split("/")[-1].split(".")[0]
    name = f"{outfile}_ts_out.xyz"
    newpath = os.path.join(directory,name)
    with open(newpath,'w') as f:
        f.write(str(num_atoms))
        f.write("\n")
        f.write(f"Energy: {energy} Final Frequencies: {data[freq_line]}")
        for l in final_structure:
            parts = l.split()
            for x in parts[1:]:
                f.write(x)
                f.write('\t')
            f.write('\n')

if __name__ == "__main__":
    process_tsoutfile(sys.argv[1])