PYTHON_SCRIPT="FFD.py"

for dataset in Cora Citeseer Photo cornell wisconsin texas
#for dataset in Citeseer Photo wisconsin
do
 for max_epochs in 10000
 do
     for process_feature in hamming cos
     do
         for architecture in GCN
         do
             for a in 0.05 0.07 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.97 0.99
             do
#               for i in 1 2 3 4 5
#               do
                 echo "Running with parameters: --dataset_name $dataset --a $a --max_epochs $max_epochs --process_feature $process_feature --architecture $architecture --K0_mul 0.5"
                 python $PYTHON_SCRIPT --dataset_name $dataset --a $a --max_epochs $max_epochs --process_feature $process_feature --architecture $architecture --K0_mul 0.5
                 echo "-----------------------------------------"
#               done
             done
         done
     done
 done
done

