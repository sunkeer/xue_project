# xue_project   #sobol analysis

# start
(Option)C:/ProgramData/miniconda3/Scripts/activate.bat

conda create -n xue python=3.10

conda activate xue

pip install pandas numpy scikit-learn SALib matplotlib -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

python sobol_analysis.py
