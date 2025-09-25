# Static Timing Analysis (STA) Project

## Project Overview
This project performs Static Timing Analysis (STA) on a digital circuit defined in a .bench file using a Look-Up Table (LUT) for delay calculations.

## File Descriptions
- **main.py**: The main script to run STA analysis.
- **circuit.py**: Parses and constructs the circuit graph from a .bench file.
- **lut.py**: Handles Look-Up Table (LUT) operations for delay interpolation.
- **sta.py**: Implements forward and backward timing propagation.
- **requirements.txt**: Lists dependencies required to run the project.
- **README.md**: Instructions on setup and execution.

## Installation and Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
## Run this code using the line below
- python main.py --bench <path_to_bench_file> --nldm <path_to_nldm_file>