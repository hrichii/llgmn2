import csv
from llgmn import *


##1 データ読み込み
l_learn_in=[]
l_learn_out=[]
l_test_in=[]
l_test_out=[]
with open("./input_data/lea_sig.csv", "r", encoding="utf-8") as f_learn_in:
    with open("./input_data/lea_T_sig.csv", "r", encoding="utf-8") as f_learn_out:
        with open("./input_data/dis_T_sig.csv", "r", encoding="utf-8") as f_test_in:
            with open("./input_data/dis_T_sig.csv", "r", encoding="utf-8") as f_test_out:
                o_learn_in = csv.reader(f_learn_in)
                o_learn_out = csv.reader(f_learn_out)
                o_test_in = csv.reader(f_test_in)
                o_test_out = csv.reader(f_test_out)
                l_learn_in=[list(map(float,l_row)) for l_row in o_learn_in]
                l_learn_out=[list(map(float,l_row)) for l_row in o_learn_out]
                l_test_in=[list(map(float,l_row)) for l_row in o_test_in]
                l_test_out=[list(map(float,l_row)) for l_row in o_test_out]
                
                
##2 パラメータ設定
i_Unit_size_input=len(l_learn_in[1])                   #次元数(入力ユニット数)
i_Unit_size_output=4                                   #クラス数(出力ユニット数)
i_Model_component_size=2                               #コンポーネント数
    

##3 インスタンス化
llgmn=LLGMN(i_Unit_size_input,i_Model_component_size,i_Unit_size_output)
x,weight=llgmn.nonlinear(l_learn_in[0][:])
    
##4 訓練
unit=llgmn.train(x)
    
    
##5 検証
    
    
    
##6 結果の出力
    
    
    