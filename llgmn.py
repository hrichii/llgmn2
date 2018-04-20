import numpy
import math
import copy
from matplotlib import pyplot

class LLGMN:
    
    def __init__(self,i_Unit_size_input, i_Model_component_size, i_Unit_size_output, i_Batch_size,Epoch):
        self.i_D=i_Unit_size_input
        self.i_H=int(1+i_Unit_size_input*(i_Unit_size_input+3)/2)
        self.i_M=i_Model_component_size
        self.i_C=i_Unit_size_output
        self.i_B=i_Batch_size
        self.weight=[[numpy.random.random_sample((self.i_M, self.i_H))] for l in range(1) for c in range(self.i_C)]
        self.error=numpy.zeros(Epoch)
    def __nonlinear(self, l_one_data_in):
        D=self.i_D
        x_pre=numpy.array(l_one_data_in)
        x=numpy.zeros((self.i_H,1))
        #i=0
        x[0,]=1
        #i=1~4
        x[1:D+1,]=x_pre.reshape(-1,1)
        i=D+1
        for left in range(D):
            for right in range(left,D):
                x[i,]=x_pre[left]*x_pre[right]
                i+=1
        return x
    
    def train(self, l_one_data_in, t, Mu, Epochnum):
        unit=self.forward(l_one_data_in)
        ###
        t=numpy.array(t)
        t.reshape(-1,1)
        for c in range(self.i_C):
            self.error[Epochnum]+=1/2*(unit[c][2]["out"]-t[c])**2
        
        ###
        self.__update_weight(Mu,unit,t)
        #return unit
        
    def forward(self, l_one_data_in):
        x=self.__nonlinear(l_one_data_in)
        unit=self.__unitx(x)
        
        sumexp=0
        #0out-1in
        for c in range(self.i_C):
            layer=0
            layer_front=layer+1
            unit[c][layer_front]["in"]=numpy.dot(self.weight[c][layer], unit[c][layer]["out"])
            for m in unit[c][layer_front]["in"]:
                sumexp+=math.exp(m)
        #1in-1out
        for c in range(self.i_C):
            layer=1
            for m in range(unit[c][layer]["in"].shape[0]):
                unit[c][layer]["out"][m,]=math.exp(unit[c][layer]["in"][m,])/sumexp
                
        #1out-2in
        for c in range(self.i_C):
            layer=1
            layer_front=layer+1
            unit[c][layer_front]["in"]=numpy.dot(numpy.ones((unit[c][layer_front]["in"].shape[0],unit[c][layer]["out"].shape[0])), unit[c][layer]["out"])
    
        #2in-2out
        for c in range(self.i_C):
            layer=2
            unit[c][layer]["out"]=unit[c][layer]["in"]
        
        return unit
        
    def __update_weight(self,Mu,unit,t):
        for c in range(self.i_C):
            delta=numpy.dot((unit[c][1]["out"]-unit[c][1]["out"]/unit[c][2]["out"]*t[c]),unit[c][0]["out"].T)
            self.weight[c][0]-=Mu*delta

        
        # 1-2 誤り改善状況の出力
    def error_graph(self):
        print("\n\n\n")
        print("========================(2)誤り改善状況========================")
        #pyplot.ylim(0.0, 2.0)
        pyplot.plot(numpy.arange(0, self.error.shape[0]), self.error)
        pyplot.show()
        print("=============================================================")
    
    
    def __unitx(self,x):
        units_0layer_in=x
        units_0layer_out=x
        units_0layer={"in":units_0layer_in,"out":units_0layer_out}
        
        units_1layer_in=numpy.zeros((self.i_M,1))#####
        units_1layer_out=units_1layer_in
        units_1layer={"in":units_1layer_in,"out":units_1layer_out}
        
        units_2layer_in=numpy.zeros((1,1))#####
        units_2layer_out=units_2layer_in
        units_2layer={"in":units_2layer_in,"out":units_2layer_out}
        
        unit0=[[copy.deepcopy(units_0layer),copy.deepcopy(units_1layer),copy.deepcopy(units_2layer)] for i in range(self.i_C)]
        return unit0
    ## 0 コンストラクタ
#    def __init__(self):

#
#        
#    ### public
#    ## 1 学習
#    def train(self, datas_input, datas_output_true, Epsilon, Mu, Epoch, Online):
#        
#        self.error = numpy.zeros(Epoch)
#        for Count_epo in range(Epoch):
#            if Online==1:
#                for iy in range(datas_input.shape[0]):
#                    onedata_input=datas_input[iy, :].reshape(1,-1)
#                    onedata_output_true=datas_output_true[iy, :].reshape(1,-1)
#                    err,datas_unit = self.__update_weight(onedata_input, onedata_output_true, Epsilon, Mu)
#                    self.error[Count_epo]+=err
#            else:
#                self.error[Count_epo],datas_unit = self.__update_weight(datas_input, datas_output_true, Epsilon, Mu)
#            '''    
#            if (Count_epo+1) % (Epoch/10) ==0:
#                print("学習率：",(Count_epo+1)*100 / Epoch,"%")
#                print("誤差",self.error[Count_epo],end="\n\n")
#            '''
#        self.__error_graph()
#        return datas_unit,self.weight
#        
#    
#    ## 2 検証
#    def predict(self, datas_input):
#        DataSize = datas_input.shape[0]
#        unit0=[numpy.zeros((self.unit_size[i],1)) for i in range(len(self.unit_size))]
#        datas_unit = [unit0 for i in range(DataSize)]
#        #print(datas_unit)
#        for iy in range(DataSize):
#            onedata_input=datas_input[iy, :].reshape(1,-1)
#            onedata_unit = self.__forward(iy,onedata_input)# 1-1-1
#            datas_unit[iy] = onedata_unit
#
#        return datas_unit#(C, Y)
#
#
#    ### private
#    # 1-1 重みの更新 : 誤差逆伝搬法を用いている
#    def __update_weight(self, datas_input, datas_output_true, Epsilon, Mu):
#        DataSize = datas_input.shape[0]# =1
#        err = 0.0
#        unit0=[numpy.zeros((self.unit_size[i],1)) for i in range(len(self.unit_size))]
#        datas_unit = [unit0 for i in range(DataSize)]
#        
#        # (1) 重みと入力から各ユニットの値を計算
#        for iy in range(DataSize):
#            onedata_input=datas_input[iy, :].reshape(1,-1)
#            onedata_output_true=datas_output_true[iy,:].reshape(1,-1)
#            onedata_unit = self.__forward(iy,onedata_input)# 1-1-1
#            datas_unit[iy]=onedata_unit
#            onedata_output=onedata_unit[len(self.unit_size)-1]
#            err += numpy.dot( (onedata_output_true.reshape(1,-1) - onedata_output.reshape(1,-1)).reshape(1,-1), (onedata_output_true.reshape(1,-1) - onedata_output.reshape(1,-1)).reshape((-1, 1)) ) / 2.0
#        # (2) 次の層
#        olddelta=[[0] for i in range(DataSize)]
#        for Count_layer in range(len(self.unit_size)-2,-1,-1):#4,3,2,1,0,
#            sumdelta=numpy.zeros((self.weight[Count_layer].shape[0],self.weight[Count_layer].shape[1]))
#            oldweight=self.weight[Count_layer]
#            for Count_data in range(DataSize):
#                #print(Count_data,"番目")
#                if Count_layer==(len(self.unit_size)-2):#5
#                    newdelta = (datas_unit[Count_data][Count_layer+1] - datas_output_true[Count_data, :].reshape(-1,1) )    * datas_unit[Count_data][Count_layer+1] * (1.0 - datas_unit[Count_data][Count_layer+1])
#                else:
#                    newdelta = numpy.dot(self.weight[Count_layer+1][:, 1:].T, olddelta[Count_data])                         * datas_unit[Count_data][Count_layer+1] * (1.0 - datas_unit[Count_data][Count_layer+1])
#                sumdelta += newdelta.reshape((-1, 1)) * (numpy.r_[numpy.ones((1,1)), datas_unit[Count_data][Count_layer]]).T
#                #self.weight[Count_layer] -= Epsilon * (newdelta.reshape((-1, 1)) * (numpy.r_[numpy.ones((1,1)), datas_unit[Count_data][Count_layer]]).T)
#                #if Count_layer==2 :
#                #    print(self.weight[Count_layer][0,0])
#                olddelta[Count_data] = newdelta
#
#                self.weight[Count_layer] -= Epsilon *sumdelta/DataSize #-0.9*self.momentum[Count_layer]
#            
#            self.momentum[Count_layer]=self.weight[Count_layer]-oldweight
#        return err,datas_unit
#    
#    
#    # 1-1-1 フォワード : 与えられた w,bを計算し,シグモイド関数を適応することで、各ユニット(入力,隠れ,出力ユニット)を算出する
#    def __forward(self,DataNum , onedata_input):
#        onedata_unit=[numpy.zeros((self.unit_size[i],1)) for i in range(len(self.unit_size))];
#        onedata_unit[0]=onedata_input.reshape(-1,1)
#        for Count_layer in range(len(self.unit_size)-1):#012345
#            onedata_unit[Count_layer+1] = self.__sigmoid(   numpy.dot( self.weight[Count_layer], numpy.concatenate([numpy.ones((1,1)), onedata_unit[Count_layer]], axis=0) )   )
#        return onedata_unit
#    
#    
#    # 1-1-1-1 シグモイド関数
#    def __sigmoid(self, arr):
#        return numpy.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x)))(arr)
#    
#    
#    # 1-2 誤り改善状況の出力
#    def __error_graph(self):
#        print("\n\n\n")
#        print("========================(2)誤り改善状況========================")
#        pyplot.ylim(0.0, 2.0)
#        pyplot.plot(numpy.arange(0, self.error.shape[0]), self.error)
#        pyplot.show()
#        print("=============================================================")
