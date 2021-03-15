<h1> Instruction </h1>

1. Connect to narvi cluster [narvi cluster](https://tuni-itc.github.io/wiki/Technical-Notes/tuni-narvi-cluster/)
2. Open the terminal and type all command in the file needed_modules.txt
3. Remember to always use the virtual environment torch-dcase that have just created to run the scripts. You can do so by running the following command `conda activate torch-dcase`.
4. Extract all features by running the file extract_feature.py `python extract_feature.py`
5. Then run the training script `training.py` using the following command: `srun     --pty     --job-name pepe_run     --partition gpu     --gres gpu:2     --mem=16G     --ntasks --cpus-per-task 10     --time 05:30:00     python training.py` or by running the bash file `run.sh` <br>

**Note:** 
The above instruction is for those have the access right to the [narvi cluster](https://tuni-itc.github.io/wiki/Technical-Notes/tuni-narvi-cluster/) and its gpu group. If you can access to other kinds of gpus, then skip step 1 and at step 4, using whatever command to run the file `training.py.
