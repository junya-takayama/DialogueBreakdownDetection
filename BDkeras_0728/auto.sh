#!/bin/bash
mkdir $2

for i in `seq 1 $1`
do
    echo "学習開始 $i/$1"
    python3 BDkeras.py $i test$i
    echo "学習終了 $i/$1"
    echo "評価用ファイル出力"
    python3 evaluate.py ./models/test${i}.pickle
    echo "評価結果出力中"
    for domain in DCM DIT IRS
    do
	python eval.py -p ../DBDC2_ref/${domain} -o ./result/${domain} -t 0.5 > ./${2}/${domain}_${1}.txt
    done
    echo "プロセス終了 $i/$1"
done

echo "全プロセス終了"