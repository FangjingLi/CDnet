import subprocess


#  Define the Python program path to run
PYTHON_SCRIPT = "CGDC.py"

# Define parameter list
PARAMETER_SETS = [
# # GCN
#     {"dataset_name": "Cora", "a": "0.99", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "hamming","K0_mul": "0.5"},
#     {"dataset_name": "Citeseer", "a": "0.6", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
#     {"dataset_name": "Photo", "a": "0.99", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
#     {"dataset_name": "squirrel", "a": "0.6", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
#     {"dataset_name": "chameleon", "a": "0.6", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
#     {"dataset_name": "actor", "a": "0.05", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
#     {"dataset_name": "cornell", "a": "0.07", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
#     {"dataset_name": "wisconsin", "a": "0.05", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
#     {"dataset_name": "texas", "a": "0.45", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
#
#     {"dataset_name": "Cora", "a": "0.97", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
#     {"dataset_name": "Citeseer", "a": "0.99", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
#     {"dataset_name": "Photo", "a": "0.99", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
#     {"dataset_name": "squirrel", "a": "0.95", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
#     {"dataset_name": "chameleon", "a": "0.99", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
#     {"dataset_name": "actor", "a": "0.99", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
#     {"dataset_name": "cornell", "a": "0.95", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
#     {"dataset_name": "wisconsin", "a": "0.1", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
#     {"dataset_name": "texas", "a": "0.1", "max_epochs": "10000", "architecture": "GCN", "calcu_weak": "True",
#      "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},

# GAT
    {"dataset_name": "Cora", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "Citeseer", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "Photo", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "squirrel", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "chameleon", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "actor", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "cornell", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "wisconsin", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "texas", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},

    {"dataset_name": "Cora", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "Citeseer", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "Photo", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "squirrel", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "chameleon", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "actor", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "cornell", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "wisconsin", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "texas", "a": "1", "max_epochs": "10000", "architecture": "GAT", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},

# GraphSAGE
    {"dataset_name": "Cora", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "Citeseer", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "Photo", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "squirrel", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "chameleon", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "actor", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "cornell", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "wisconsin", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},
    {"dataset_name": "texas", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "hamming", "K0_mul": "0.5"},

    {"dataset_name": "Cora", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "Citeseer", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "Photo", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "squirrel", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "chameleon", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "actor", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "cornell", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "wisconsin", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
    {"dataset_name": "texas", "a": "1", "max_epochs": "10000", "architecture": "GraphSAGE", "calcu_weak": "True",
     "use_lcc": "True", "process_feature": "cos", "K0_mul": "0.5"},
]

# Loop through the parameter list
for parameters in PARAMETER_SETS:
    command = ["python", PYTHON_SCRIPT]
    for key, value in parameters.items():
        command.extend(["--" + key, str(value)])

    print(f"Running with parameters: {' '.join(command[2:])}")
    subprocess.run(command)
    print("-----------------------------------------")
