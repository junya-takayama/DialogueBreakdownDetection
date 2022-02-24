# echo "CNN_single"
# python predict.py -k kmeans_cnn -n 1 --data ./data/maxpmi/data_undersampled.csv 
# echo "LSTM_single"
# python predict.py -c kmeans_lstm -m 1 --data ./data/maxpmi/data_undersampled.csv
echo "CNN:5"
python predict.py -k kmeans_cnn -n 5 --data ./data/maxpmi/data_undersampled.csv --output_path ./rg_results/cnn_5.csv
echo "LSTM:5"
python predict.py -c kmeans_lstm -m 5 --data ./data/maxpmi/data_undersampled.csv --output_path ./rg_results/lstm_5.csv
echo "CNN+LSTM:5"
python predict.py -c kmeans_lstm -m 5 -k kmeans_cnn -n 5 --data ./data/maxpmi/data_undersampled.csv --output_path ./rg_results/ensemble.csv
# echo "CNN_random:7"
# python predict.py -k random_cnn -n 7 --data ./data/maxpmi/data_undersampled.csv
# echo "LSTM_random:10"
# python predict.py -c random_lstm -m 10 --data ./data/maxpmi/data_undersampled.csv