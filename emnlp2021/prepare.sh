mkdir -p models

# DOWNLOAD THE MODELS FOR EVALUATION
# download the CoLA model from http://style.cs.umass.edu/
# using several runs because gdown does not preserve folders
mkdir -p models/cola
mkdir -p models/cola/cola-bin
gdown https://drive.google.com/drive/folders/18G5ZfLRKTMlV0Ke4shztUlGZ6H8j8AD6 -O  models/cola/cola-bin/input0 --folder
gdown https://drive.google.com/drive/folders/1UlU0g9HUerK0xW8B_MKV08DkIPEs44-Q -O  models/cola/cola-bin/label --folder
gdown --id 1rBSrbL_6gfDqOCpNz1JrOJK9gKrkKxhv -O models/cola/checkpoint_best.pt
# download the similarity model from http://style.cs.umass.edu/
gdown https://drive.google.com/drive/folders/1lBN2nbzxtpqbPUyeURtzt0k1kBY6u6Mj -O models/sim --folder

# DOWNLOAD THE MODELS FOR INFERENCE

