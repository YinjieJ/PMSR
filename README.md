Run PMSR using these command:


```
cd PMSR
bash script/preprocess.sh
bash script/finetune.sh 0 # the argument is the id of visible GPU
bash script/generate.sh
```