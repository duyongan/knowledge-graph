# coding: utf-8
import numpy as np
import math
import random


#负样本打乱器1
def gx_to_find_en(p):
    gx=p
    while gx==p:
        triple=random.choice(triples)
        p=triple[1]
    return triple[0],triple[2]
	
	
#负样本打乱器2
def en_to_find_gx(p):
    gx=p
    while gx==p:
        p=random.choice(relations)
    return p  


#能量函数
def energe_function(miu_e,miu_r,sigma_e,sigma_r,k_e):
    miu_T=(miu_e-miu_r).T
    sigma_I=(sigma_e+sigma_r).I
    result=miu_T*sigma_I
    result=result*(miu_e-miu_r)
    x=sigma_e+sigma_r
    x=np.linalg.det(x)
    x=np.log(x)
    result=result+x
    result=result+k_e*np.log(2*math.pi)
    return 0.5*result


#初始化随机期望与协方差矩阵
def init_miu_or_sigma(k_e,sigma_min,sigma_max):
    miu = min(np.random.uniform(-6/np.sqrt(k_e),6/np.sqrt(k_e)),1)
    sigma_min = max([0,sigma_min])
    sigma = np.random.uniform(sigma_min,sigma_max)
    sigma_max = min([sigma,sigma_max])
    return np.mat(miu),np.mat(sigma)


#根据头和尾 得到实体期望和协方差矩阵
def get_miu_e_or_sigma_e(miu_h,miu_t,sigma_h,sigma_t):
    miu_e = miu_h-miu_t
    sigma_e=sigma_h+sigma_t
    return miu_e,sigma_e
	

#得到梯度
def get_gradient(miu_h,miu_r,miu_t,sigma_h,sigma_r,sigma_t):
    sigma_prime = sigma_h+sigma_r+sigma_t
    theta_prime = sigma_prime.I*(miu_r+miu_t-miu_h)
    sigma_gradient=theta_prime*theta_prime.T-sigma_prime.I
    return theta_prime,sigma_gradient


def train(entities,relations,triples,k_e,sigma_min,sigma_max,lr,epoch,batch_size,lamda):
    if len(triples)%batch_size==0:
        if_left=False
        loop=int(len(triples)/batch_size)
    else:
        if_left=True
        loop=int(len(triples)/batch_size)+1
    en_re_map={}
    for l in entities+relations:
        miu,sigma=init_miu_or_sigma(k_e,sigma_min,sigma_max)
        en_re_map[l]=[miu,sigma]
    for j in range(epoch):
        sum_loss=0
        for ii in range(loop):
            if if_left and ii==loop-1:
                batch=len(triples)-ii*batch_size
            else:
                batch=batch_size
            batch_loss=0
            for i in range(batch):
                #初始化正样本与负样本
                triple=triples[ii*batch_size+i]
                gx2=en_to_find_gx(triple[1])
                nagative_triple1 = [en_re_map[triple[0]],en_re_map[gx2],en_re_map[triple[2]]]
                temp=gx_to_find_en(triple[1])
                nagative_triple2 = [en_re_map[temp[0]],en_re_map[triple[1]],en_re_map[temp[1]]]
                triple_vec=[en_re_map[triple[0]],en_re_map[triple[1]],en_re_map[triple[2]]]


                #第一个负样本
                #能量函数计算
                miu_e1,sigma_e1 = get_miu_e_or_sigma_e(nagative_triple1[0][0],nagative_triple1[2][0],
                                                       nagative_triple1[0][1],nagative_triple1[2][1])
                energe1 = energe_function(miu_e1,nagative_triple1[1][0],sigma_e1,nagative_triple1[1][1],k_e)
                #梯度上升
                miu_gradient,sigma_gradient=get_gradient(nagative_triple1[0][0],nagative_triple1[1][0],
                                    nagative_triple1[2][0],nagative_triple1[0][1],nagative_triple1[1][1],nagative_triple1[2][1])
                en_re_map[triple[0]][0]=np.mat(max(min(1,nagative_triple1[0][0]+lr*miu_gradient),-1))
                en_re_map[gx2][0]=np.mat(max(min(1,nagative_triple1[1][0]+lr*miu_gradient),-1))
                en_re_map[triple[2]][0]=np.mat(max(min(1,nagative_triple1[2][0]+lr*miu_gradient),-1))
                en_re_map[triple[0]][1]=np.mat(max(sigma_min,min(sigma_max,nagative_triple1[0][0]+lr*sigma_gradient)))
                en_re_map[gx2][1]=np.mat(max(sigma_min,min(sigma_max,nagative_triple1[1][0]+lr*sigma_gradient)))
                en_re_map[triple[2]][1]=np.mat(max(sigma_min,min(sigma_max,nagative_triple1[2][0]+lr*sigma_gradient)))       

                #第二个负样本
                #能量函数计算
                miu_e2,sigma_e2 = get_miu_e_or_sigma_e(nagative_triple2[0][0],nagative_triple2[2][0],
                                                       nagative_triple2[0][1],nagative_triple2[2][1])
                energe2 = energe_function(miu_e2,nagative_triple2[1][0],sigma_e2,nagative_triple2[1][1],k_e)
                #梯度上升
                miu_gradient,sigma_gradient=get_gradient(nagative_triple2[0][0],nagative_triple2[1][0],
                                        nagative_triple2[2][0],nagative_triple2[0][1],nagative_triple2[1][1],nagative_triple2[2][1])
                en_re_map[temp[0]][0]=np.mat(max(min(1,nagative_triple2[0][0]+lr*miu_gradient),-1))
                en_re_map[triple[1]][0]=np.mat(max(min(1,nagative_triple2[1][0]+lr*miu_gradient),-1))
                en_re_map[temp[1]][0]=np.mat(max(min(1,nagative_triple2[2][0]+lr*miu_gradient),-1))
                en_re_map[temp[0]][1]=np.mat(max(sigma_min,min(sigma_max,nagative_triple2[0][0]+lr*sigma_gradient)))
                en_re_map[triple[1]][1]=np.mat(max(sigma_min,min(sigma_max,nagative_triple2[1][0]+lr*sigma_gradient)))
                en_re_map[temp[1]][1]=np.mat(max(sigma_min,min(sigma_max,nagative_triple2[2][0]+lr*sigma_gradient)))

                #正样本
                #能量函数计算
                miu_e,sigma_e = get_miu_e_or_sigma_e(triple_vec[0][0],triple_vec[2][0],triple_vec[0][1],triple_vec[2][1])
                energe = energe_function(miu_e,triple_vec[1][0],sigma_e,triple_vec[1][1],k_e)
                #梯度下降
                miu_gradient,sigma_gradient=get_gradient(triple_vec[0][0],triple_vec[1][0],
                                                         triple_vec[2][0],triple_vec[0][1],triple_vec[1][1],triple_vec[2][1])
                en_re_map[triple[0]][0]=np.mat(max(min(1,triple_vec[0][0]-lr*miu_gradient),-1))
                en_re_map[triple[1]][0]=np.mat(max(min(1,triple_vec[1][0]-lr*miu_gradient),-1))
                en_re_map[triple[2]][0]=np.mat(max(min(1,triple_vec[2][0]-lr*miu_gradient),-1))
                en_re_map[triple[0]][1]=np.mat(max(sigma_min,min(sigma_max,triple_vec[0][0]-lr*sigma_gradient)))
                en_re_map[triple[1]][1]=np.mat(max(sigma_min,min(sigma_max,triple_vec[1][0]-lr*sigma_gradient)))
                en_re_map[triple[2]][1]=np.mat(max(sigma_min,min(sigma_max,triple_vec[2][0]-lr*sigma_gradient)))
                batch_loss=batch_loss+(max(0,float(np.array(energe)[0][0]-np.array(energe2)[0][0]+lamda))
                +max(0,float(np.array(energe)[0][0]-np.array(energe1)[0][0]+lamda)))/2
            sum_loss=sum_loss+batch_loss
        print('第'+str(j)+'训练  平均损失为：'+str(float(sum_loss/len(triples))))
    return en_re_map

