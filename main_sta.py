import argparse
import numpy as np
from collections import deque

class Node:
    def __init__(self, name, gate_type):
        self.name = name
        self.gate_type = gate_type
        self.inputs = []
        self.outputs = []
        self.Cload = 0.0
        self.Tau_in = 0.002  
        self.outp_arrival = 0.0
        self.required_arrival = float('inf')
        self.slack = 0.0

class Circuit:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.gates = {}

    def parse_bench(self, filepath):
        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "INPUT" in line:
                    node_name = line.split("(")[1].strip(")")
                    self.inputs.append(node_name)
                    self.gates[node_name] = Node(node_name, "INPUT")
                elif "OUTPUT" in line:
                    node_name = line.split("(")[1].strip(")")
                    self.outputs.append(node_name)
                    if node_name not in self.gates:
                        self.gates[node_name] = Node(node_name, "OUTPUT")
                else:
                    parts = line.split("=")
                    output_node = parts[0].strip()
                    gate_info = parts[1].strip().split("(")
                    gate_type = gate_info[0].strip()
                    inputs = gate_info[1].strip(")").split(",")
                    node = Node(output_node, gate_type)
                    node.inputs = [inp.strip() for inp in inputs]
                    self.gates[output_node] = node
                    for inp in node.inputs:
                        if inp not in self.gates:
                            self.gates[inp] = Node(inp, "WIRE")
                        self.gates[inp].outputs.append(output_node)
 
    
            
class LUT:
    def __init__(self):
        self.gate = ''
        self.all_del = []
        self.all_slew = []
        self.cload_val = []
        self.Tau_in_val = []
        self.cap = 0

class Cells:
    def __init__(self):
        self.cells = []

    def assign_arrays(self, NLDM_file):
        text = open(NLDM_file).read()
        index = text.find('}')
        text = text[index + 1:]
        
        lines = text.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]
            
            if 'cell ' in line:
                new_cell = LUT()
                new_cell.gate = line.split('(')[1].split(')')[0]
                
                while 'capacitance' not in line:
                    i+=1
                    line = lines[i]
                new_cell.cap = float(line.split(':')[1].strip(';'))    
                while 'cell_delay' not in line:
                    i += 1
                    line = lines[i]

                i += 1
                line = lines[i]

                new_cell.Tau_in_val = line.split('(')[1].split(')')[0][1:-1].split(',')

                i += 1
                line = lines[i]

                new_cell.cload_val = line.split('(')[1].split(')')[0][1:-1].split(',')

                self.cells.append(new_cell)

                i += 1
                line = lines[i]
                while '"' in line:
                    values = line.split('"')[1].split(',')
                    new_cell.all_del.append(values)

                    i += 1
                    line = lines[i]

                while 'values' not in line:
                    i += 1
                    line = lines[i]

                while '"' in line:
                    values = line.split('"')[1].split(',')
                    new_cell.all_slew.append(values)

                    i += 1
                    line = lines[i]
                
            i += 1 

   


class STA:
    def __init__(self, circuit, lut):
        self.circuit = circuit
        self.lut = lut
        self.circuit_delay = 0.0
        self.slacks = {}
        self.critical_path = []

    def isinterpolation(self, slew, load, cell, table_type='delay'):
        slew_vals = np.array([float(x) for x in cell.Tau_in_val])
        load_vals = np.array([float(x) for x in cell.cload_val])
        table = cell.all_del if table_type == 'delay' else cell.all_slew

        
        if slew in slew_vals and load in load_vals:
            slew_idx = np.where(slew_vals == slew)[0][0]
            load_idx = np.where(load_vals == load)[0][0]
            return float(table[slew_idx][load_idx])

        
        if slew >= slew_vals[-1]:  
            p = len(slew_vals) - 2
            q = len(slew_vals) - 1
        elif slew <= slew_vals[0]:  
            p = 0
            q = 1
        else:  
            p = np.searchsorted(slew_vals, slew) - 1
            q = p + 1

        if load >= load_vals[-1]:  
            r = len(load_vals) - 2
            s = len(load_vals) - 1
        elif load <= load_vals[0]:  
            r = 0
            s = 1
        else:  
            r = np.searchsorted(load_vals, load) - 1
            s = r + 1



        tau1, tau2 = slew_vals[p], slew_vals[q]
        c1, c2 = load_vals[r], load_vals[s]

        v11 = float(table[p][r])
        v12 = float(table[p][s])
        v21 = float(table[q][r])
        v22 = float(table[q][s])

        
        interpolated_val = (
            v11 * (tau2 - slew) * (c2 - load) +
            v12 * (tau2 - slew) * (load - c1) +
            v21 * (slew - tau1) * (c2 - load) +
            v22 * (slew - tau1) * (load - c1)
        ) / ((tau2 - tau1) * (c2 - c1))

        return interpolated_val
    
    def bfs_traversal(self):
        visited = set() 
        bfs_order = []
        queue = deque(self.circuit.inputs[:])  

        while queue:
            node_name = queue.popleft()

            curr_node = self.circuit.gates[node_name]

            if any(input_node not in visited for input_node in curr_node.inputs):
                for input_node in curr_node.inputs:
                    if input_node not in visited:
                        queue.append(input_node)
                queue.append(node_name)  
                continue  

            if node_name not in visited:
                visited.add(node_name)
                bfs_order.append(node_name)


                for out_node in curr_node.outputs:
                    if out_node not in visited:
                        queue.append(out_node)
        return bfs_order

    def forward_traversal(self):
        inverter_capacitance = 1.700230

        for gate_name, gate in sorted(self.circuit.gates.items(), key=lambda x: x[0], reverse=True):
            gate.Cload = 0.0

            if gate_name in self.circuit.outputs:
                gate.Cload = 4 * inverter_capacitance

            if gate.gate_type != "INPUT":
                for out in gate.outputs:
                        out_gate_type = self.circuit.gates[out].gate_type  

                        
                        if out_gate_type == "BUFF":
                            out_lut_index = 6
                        elif out_gate_type == "NOT":
                            out_lut_index = 5
                        elif out_gate_type == "XOR":
                            out_lut_index = 4
                        elif out_gate_type == "OR":
                            out_lut_index = 3
                        elif out_gate_type == "AND":
                            out_lut_index = 2
                        elif out_gate_type == "NOR":
                            out_lut_index = 1
                        elif out_gate_type == "NAND":
                            out_lut_index = 0
                        else:
                            raise ValueError(f"Unknown output gate type: {out_gate_type}")
                       
                        gate.Cload += self.lut.cells[out_lut_index].cap
                    

        for node in self.bfs_traversal():
            fanout_node = self.circuit.gates[node]                      
            
            max_arrival = 0.0
            max_slew = 0.0
            for inp_name in fanout_node.inputs:
                inp_node = self.circuit.gates[inp_name]
                if fanout_node.gate_type == "BUFF":
                    lut_index = 6
                elif fanout_node.gate_type == "NOT":
                    lut_index = 5
                elif fanout_node.gate_type == "XOR":
                    lut_index = 4
                elif fanout_node.gate_type == "OR":
                    lut_index = 3
                elif fanout_node.gate_type == "AND":
                    lut_index = 2
                elif fanout_node.gate_type == "NOR":
                    lut_index = 1
                elif fanout_node.gate_type == "NAND":
                    lut_index = 0
                else:
                    raise ValueError(f"Unknown gate type: {node.gate_type}")               
                delay = self.isinterpolation(inp_node.Tau_in, fanout_node.Cload, self.lut.cells[lut_index], 'delay')
                if (len(fanout_node.inputs) > 2):
                    delay = delay * len(fanout_node.inputs) / 2   
                if delay < 0:
                    delay = 0                                 
                arrival_time = inp_node.outp_arrival + delay
                if arrival_time > max_arrival:
                    max_arrival = arrival_time
                    fanout_node.outp_arrival = max_arrival
                    max_slew = self.isinterpolation(inp_node.Tau_in, fanout_node.Cload, self.lut.cells[lut_index], 'slew')
                    if (len(fanout_node.inputs) > 2):
                        max_slew = max_slew * len(fanout_node.inputs) / 2 
                    if max_slew < 0.002:
                        max_slew = 0.002 
                    fanout_node.Tau_in = max_slew

        self.circuit_delay = max(self.circuit.gates[out].outp_arrival for out in self.circuit.outputs)

    def backward_traversal(self):
        required_time = 1.1 * self.circuit_delay

        for out in self.circuit.outputs:
            self.circuit.gates[out].required_arrival = required_time
            self.slacks[f"OUTPUT-{out}"] = required_time - self.circuit.gates[out].outp_arrival

        reverse_order = reversed(self.bfs_traversal())

        for node_name in reverse_order:
            node = self.circuit.gates[node_name]
            for input_name in node.inputs:
                input_node = self.circuit.gates[input_name]
                if node.gate_type == "BUFF":
                    lut_index = 6
                elif node.gate_type == "NOT":
                    lut_index = 5
                elif node.gate_type == "XOR":
                    lut_index = 4
                elif node.gate_type == "OR":
                    lut_index = 3
                elif node.gate_type == "AND":
                    lut_index = 2
                elif node.gate_type == "NOR":
                    lut_index = 1
                elif node.gate_type == "NAND":
                    lut_index = 0
                else:
                    raise ValueError(f"Unknown gate type: {node.gate_type}")
                deli = self.isinterpolation(input_node.Tau_in, node.Cload, self.lut.cells[lut_index], 'delay')
                if deli < 0:
                    deli = 0   
                required_at_input = node.required_arrival - deli                                 

                if required_at_input < input_node.required_arrival:
                    input_node.required_arrival = required_at_input

                self.slacks[f"{input_node.gate_type}-{input_name}"] = (
                    input_node.required_arrival - input_node.outp_arrival)

    def find_critical_path(self):
        min_slack_out = min(self.circuit.outputs, key=lambda o: self.slacks[f"OUTPUT-{o}"])
        output_gate_type = self.circuit.gates[min_slack_out].gate_type  
        path = [f"OUTPUT-{min_slack_out}"]
        current = min_slack_out

        while current not in self.circuit.inputs:
            node = self.circuit.gates[current]
            min_input = min(node.inputs, key=lambda i: self.slacks[f"{self.circuit.gates[i].gate_type}-{i}"])
            path.append(f"{self.circuit.gates[min_input].gate_type}-{min_input}")
            current = min_input

        path = path[::-1]
        path.insert(-1, f"{output_gate_type}-{min_slack_out}")

        return path

    def run_sta(self):
        self.bfs_traversal()
        self.forward_traversal()
        self.backward_traversal()
        self.critical_path = self.find_critical_path()
        

    def write_results(self, filepath):
        with open(filepath, 'w') as file:
            file.write(f"Circuit delay: {self.circuit_delay*1e3:.4f}ps\n")
            file.write("Gate slacks:\n")
            for gate in sorted(self.slacks):
                file.write(f"{gate}: {self.slacks[gate]*1e3:.4f} ps\n")
            file.write("\nCritical path:\n")
            file.write(", ".join(self.critical_path))

def main():
    parser = argparse.ArgumentParser(description="Perform STA analysis.")
    parser.add_argument("--read_ckt", required=True, help="Path to the bench file.")
    parser.add_argument("--read_nldm", required=True, help="Path to the NLDM liberty file.")
    
   

    args = parser.parse_args()

    circuit = Circuit()
    circuit.parse_bench(args.read_ckt)
    

    lut = Cells()
    lut.assign_arrays(args.read_nldm)

   
    sta = STA(circuit, lut)
    sta.run_sta()
    sta.write_results("ckt_traversal.txt")
    print("STA completed, results saved to ckt_traversal.txt")

if __name__ == "__main__":
    main()