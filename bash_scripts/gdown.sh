# remember do not place space between variable and equal sign
# please modify 
gdown_id="1_ki7Jo00jO1utKEpQkQ4GChD4iwZpWS8"
file_name="240926124017.zip"
# meanwhile please modify configs/iphone/nerfcapture_off.py
# scene_name and frame_num

experiment_path="./experiments/iPhone_Captures"
cd $experiment_path
gdown $gdown_id
unzip $file_name
cd ../../
python scripts/offline_iphone.py $file_name --config configs/iphone/nerfcapture_off.py
python scripts/splatam.py 
python scripts/export_ply.py