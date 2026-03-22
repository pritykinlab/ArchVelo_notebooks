# ArchVelo_notebooks
This repository contains notebooks for generating figures in the ArchVelo paper.

ArchVelo (https://github.com/pritykinlab/ArchVelo) is a method for modeling gene regulation and inferring cell trajectories using single-cell simultaneous chromatin accessibility (scATAC-seq) and transcriptomic (scRNA-seq) profiling. ArchVelo represents chromatin accessibility as a set of archetypes---shared regulatory programs---and models their dynamic influence on transcription. As a result, ArchVelo improves inference accuracy compared to previous methods and decomposes the velocity field into components, each potentially corresponding to a specific regulatory program.

In this repository, we demonstrate ArchVelo on 3 different single-cell multi-omic datasets: mouse embryonic brain, human hematopoietic stem cells and CD8 T cells responding to acute and chronic viral infections. The notebooks for every dataset are numbered according to their order in the pipeline. Details of the required data downloads can be found in the description of each notebook. The current core structure of every folder after downloading required raw data will be as follows:

|--**Dataset_name** \
|&emsp; |-- **data**: raw data that requires preprocessing \
|&emsp; |-- **processed_data**: processed data to run the notebooks \
|&emsp; |-- **1_Data_preparation.ipynb**: process data for ArchVelo analysis and benchmarking \
|&emsp; |-- **2_Create_archetypes.ipynb**: apply archetypal analysis (AA) to the dataset \
|&emsp; |-- **3_ArchVelo.py**: apply ArchVelo to the dataset, with other options \
|&emsp; |-- **4_Compare_latent_times.ipynb**: benchmark for fit and latent time robustness \
|&emsp; |-- **5_CBDir_full_bench.ipynb**: benchmark for trajectory accuracy \
|&emsp; |-- **6_Trajectory_components.ipynb**: trajectory decomposition and interpretation of ArchVelo results \
|&emsp; |-- **7_Test_Robustness.py**: test robustness to k \
|&emsp; |-- **7_Robustness_analysis_k.ipynb**: visualization of test results \



