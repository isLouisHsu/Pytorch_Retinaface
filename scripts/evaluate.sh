cd widerface_evaluate

# python evaluation.py -p <your prediction dir> -g <groud truth dir>
python evaluation.py -p predictions/Efficientnet-b0/ -g ground_truth/
python evaluation.py -p predictions/Efficientnet-b4/ -g ground_truth/
python evaluation.py -p predictions/resnet18/ -g ground_truth/
python evaluation.py -p predictions/resnet34/ -g ground_truth/