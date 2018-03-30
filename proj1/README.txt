All the code was developed on Ubuntu 14.04 LTS. The python version used was 3.4.3. Imported libraries include pandas(0.13.1), sklearn(0.19.1), numpy(1.8.2) and matplotlib(1.3.1).

To run the analysis on dataset 1:
$ python3 model_select.py > results1.txt
To run the analysis on dataset 2:
$ python3 data2.py > results2.txt

The commands will write to "results*.txt" the best hyperparameters selected for each estimator and corresponding accuracy on the testing set generated from each best model. All of the figures included in the report will also be saved to the same directory. 
