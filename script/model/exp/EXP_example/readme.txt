This is an example folder to create a new experiment. 

1. Copy the folder "EXP_example" and put it under "exp/"
2. Rename the copied folder with the format of "Architecture_dataset_type"
3. In "hpo.py", modify lines 34-37 to parameters you want.
    basic configuration is stored at (DO NOT CHANGE)
    -> /pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/config/base.yaml
4. Run hpo.sh. Wait for it finished. 
5. Go to ./optuna and run checkHPO.ipynb. See if the total trial is 200. 
6. If successful, you will see hpo.yaml. Then run best.sh. 
7. If successful, you will see best.yaml. 
8. Analaysis codes are under 
    -> /pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/notebooks