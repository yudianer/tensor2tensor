## Problems met during learning t2t will be set here.
####  Add Problem
In addition to defining "Problem" class, add your "Problem" into **tensor2tensor/tensor2tensor/data_generators/all_problems.py** like this: 
``` 
from tensor2tensor.data_generators import translate_mnzh
``` 
Thus, t2t can find your self-defined "Problem".

Here, module **translate_mnzh.py** is set below data_generators. 

After running  ` t2t-datagen `, your "Problem" will be found from the output. 