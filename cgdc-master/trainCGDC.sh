PYTHON_SCRIPT="CGDC.py"

for dataset in Cora Citeseer Photo cornell squirrel wisconsin texas
#for dataset in Citeseer Photo wisconsin
do
 for max_epochs in 10000
 do
     for process_feature in hamming cos
     do
         for architecture in GAT GraphSAGE
         do
             for a in 1
             do
#               for i in 1 2 3 4 5
#               do
                 echo "Running with parameters: --dataset_name $dataset --a $a --max_epochs $max_epochs --process_feature $process_feature --architecture $architecture --K0_mul 7"
                 python $PYTHON_SCRIPT --dataset_name $dataset --a $a --max_epochs $max_epochs --process_feature $process_feature --architecture $architecture --K0_mul 7
                 echo "-----------------------------------------"
#               done
             done
         done
     done
 done
done

