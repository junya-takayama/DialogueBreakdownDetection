#!/bin/bash
mkdir $2

for i in `seq 1 $1`
do
    echo "学習開始 $i/$1"
    #python3 trainer.py $i $2 $3
    echo "学習終了 $i/$1"
    echo "評価用ファイル出力"
    python3 evaluate.py -n $i -k $2
    echo "評価結果出力中"
    for domain in DCM DIT IRS
    do
        python eval.py -p ./data/DBDC2_ref/${domain} -o ./result/${domain} -t 0.0 > ./${2}/${domain}_${i}.txt
    done
    echo "プロセス終了 $i/$1"
done

echo "全プロセス終了"
