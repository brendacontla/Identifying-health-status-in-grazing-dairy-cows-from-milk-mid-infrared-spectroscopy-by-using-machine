library(mltools)

## Ensemble Model

ensembleFunction<-function(y,nt,n,data,prob1,prob2,val1,val2,val3,w1,w2){
  sensi.aver1<-c()
  speci.aver1<-c()
  roc.aver1<-c()
  accu.aver1<-c()
  ppv.aver1<-c()
  npv.aver1<-c()
  mcc.aver1<-c()
  
  sensi.mv2<-c()
  speci.mv2<-c()
  roc.mv2<-c()
  accu.mv2<-c()
  ppv.mv2<-c()
  npv.mv2<-c()
  mcc.mv2<-c()
  
  sensi.we3<-c()
  speci.we3<-c()
  roc.we3<-c()
  accu.we3<-c()
  ppv.we3<-c()
  npv.we3<-c()
  mcc.we3<-c()
  b = floor(nt*n)
  
  for (i in 1:k) {
    #Data
    df<-data[[i]]
    
    if("MF" %in% colnames(df))
    {
      df<-dplyr::select(df, -MF)
    }
    
    #Data Partition
    ##Test
    df.test<-df[(b+1):n,]
    if(y=="sick"){
      t.test<-dplyr::select(df.test, -Mastitis, -Lameness)
    }else if(y=="lameness"){
      t.test<-dplyr::select(df.test, -Sick, -Mastitis)
    }else{
      t.test<-dplyr::select(df.test, -Sick, -Lameness)
    }
    
    #Ensemble Model
    
    ##Averaging
    proba.ens<-(prob1[[i]]+prob2[[i]])/2
    values.ens<-as.factor(ifelse(proba.ens>=0.5,'0','1'))
    
    ##Majority voting
    values.ensmv<-as.factor(ifelse(val1[[i]]=='0' & val2[[i]]=='0','0',ifelse(val1[[i]]=='0' & val3[[i]]=='0','0',ifelse(val2[[i]]=='0' & val3[[i]]=='0','0','1'))))
    
    #Weighted Average
    proba.wens<-(prob1[[i]]*w1)+(prob2[[i]]*w2)
    values.wens<-as.factor(ifelse(proba.wens>=0.5,'0','1'))
    
    #Specificity and Sensitivity
    
    ##Averaging
    u1 <- sort(union(values.ens, t.test[,1]), decreasing = FALSE)
    t1 <- table(factor(t.test[,1], u1), factor(values.ens, u1))
    sensi.aver1[i]<-t1[2,2]/(t1[2,1]+t1[2,2])
    speci.aver1[i]<-t1[1,1]/(t1[1,1]+t1[1,2])
    roc.aver1[i]<-as.numeric(roc.curve(t.test[,1], values.ens,plotit = F)$auc)
    accu.aver1[i]<-(t1[1,1]+t1[2,2])/(t1[1,1]+t1[1,2]+t1[2,1]+t1[2,2])
    ppv.aver1[i]<-t1[2,2]/(t1[2,2]+t1[1,2])
    npv.aver1[i]<-t1[1,1]/(t1[1,1]+t1[2,1])
    mcc.aver1[i]<-(t1[1,1]*t1[2,2]-t1[1,2]*t1[2,1])/(sqrt((t1[2,2]+t1[1,2])*(t1[2,2]+t1[2,1])*(t1[1,1]+t1[1,2])*(t1[1,1]+t1[2,1])))
    
    ##Majority voting
    u2 <- sort(union(values.ensmv, t.test[,1]), decreasing = FALSE)
    t2 <- table(factor(t.test[,1], u2), factor(values.ensmv, u2))
    sensi.mv2[i]<-t2[2,2]/(t2[2,1]+t2[2,2])
    speci.mv2[i]<-t2[1,1]/(t2[1,1]+t2[1,2])
    roc.mv2[i]<-as.numeric(roc.curve(t.test[,1], values.ensmv,plotit = F)$auc)
    accu.mv2[i]<-(t2[1,1]+t2[2,2])/(t2[1,1]+t2[1,2]+t2[2,1]+t2[2,2])
    ppv.mv2[i]<-t2[2,2]/(t2[2,2]+t2[1,2])
    npv.mv2[i]<-t2[1,1]/(t2[1,1]+t2[2,1])
    mcc.mv2[i]<-(t2[1,1]*t2[2,2]-t2[1,2]*t2[2,1])/(sqrt((t2[2,2]+t2[1,2])*(t2[2,2]+t2[2,1])*(t2[1,1]+t2[1,2])*(t2[1,1]+t2[2,1])))
    
    ##Weighted Average
    u3 <- sort(union(values.wens, t.test[,1]), decreasing = FALSE)
    t3 <- table(factor(t.test[,1], u3), factor(values.wens, u3))
    sensi.we3[i]<-t3[2,2]/(t3[2,1]+t3[2,2])
    speci.we3[i]<-t3[1,1]/(t3[1,1]+t3[1,2])
    roc.we3[i]<-as.numeric(roc.curve(t.test[,1], values.wens,plotit = F)$auc)
    accu.we3[i]<-(t3[1,1]+t3[2,2])/(t3[1,1]+t3[1,2]+t3[2,1]+t3[2,2])
    ppv.we3[i]<-t3[2,2]/(t3[2,2]+t3[1,2])
    npv.we3[i]<-t3[1,1]/(t3[1,1]+t3[2,1])
    mcc.we3[i]<-(t3[1,1]*t3[2,2]-t3[1,2]*t3[2,1])/(sqrt((t3[2,2]+t3[1,2])*(t3[2,2]+t3[2,1])*(t3[1,1]+t3[1,2])*(t3[1,1]+t3[2,1])))
  }
  
  m.sensi1<-mean(sensi.aver1, na.rm = TRUE)
  m.speci1<-mean(speci.aver1, na.rm = TRUE)
  m.roc1<-mean(roc.aver1, na.rm = TRUE)
  m.accu1<-mean(accu.aver1, na.rm = TRUE)
  m.ppv1<-mean(ppv.aver1, na.rm = TRUE)
  m.npv1<-mean(npv.aver1, na.rm = TRUE)
  m.mcc1<-mean(mcc.aver1, na.rm = TRUE)
  sd.sensi1<-sd(sensi.aver1, na.rm = TRUE)
  sd.speci1<-sd(speci.aver1, na.rm = TRUE)
  sd.roc1<-sd(roc.aver1, na.rm = TRUE)
  sd.accu1<-sd(accu.aver1, na.rm = TRUE)
  sd.ppv1<-sd(ppv.aver1, na.rm = TRUE)
  sd.npv1<-sd(npv.aver1, na.rm = TRUE)
  sd.mcc1<-sd(mcc.aver1, na.rm = TRUE)
  
  final.measure1<-c(m.sensi1,sd.sensi1,m.speci1,sd.speci1,m.accu1,sd.accu1,m.ppv1,sd.ppv1,m.npv1,sd.npv1,m.roc1,sd.roc1,m.mcc1,sd.mcc1)
  
  m.sensi2<-mean(sensi.mv2, na.rm = TRUE)
  m.speci2<-mean(speci.mv2, na.rm = TRUE)
  m.roc2<-mean(roc.mv2, na.rm = TRUE)
  m.accu2<-mean(accu.mv2, na.rm = TRUE)
  m.ppv2<-mean(ppv.mv2, na.rm = TRUE)
  m.npv2<-mean(npv.mv2, na.rm = TRUE)
  m.mcc2<-mean(mcc.mv2, na.rm = TRUE)
  sd.sensi2<-sd(sensi.mv2, na.rm = TRUE)
  sd.speci2<-sd(speci.mv2, na.rm = TRUE)
  sd.roc2<-sd(roc.mv2, na.rm = TRUE)
  sd.accu2<-sd(accu.mv2, na.rm = TRUE)
  sd.ppv2<-sd(ppv.mv2, na.rm = TRUE)
  sd.npv2<-sd(npv.mv2, na.rm = TRUE)
  sd.mcc2<-sd(mcc.mv2, na.rm = TRUE)
  
  final.measure2<-c(m.sensi2,sd.sensi2,m.speci2,sd.speci2,m.accu2,sd.accu2,m.ppv2,sd.ppv2,m.npv2,sd.npv2,m.roc2,sd.roc2,m.mcc2,sd.mcc2)
  
  m.sensi3<-mean(sensi.we3, na.rm = TRUE)
  m.speci3<-mean(speci.we3, na.rm = TRUE)
  m.roc3<-mean(roc.we3, na.rm = TRUE)
  m.accu3<-mean(accu.we3, na.rm = TRUE)
  m.ppv3<-mean(ppv.we3, na.rm = TRUE)
  m.npv3<-mean(npv.we3, na.rm = TRUE)
  m.mcc3<-mean(mcc.we3, na.rm = TRUE)
  sd.sensi3<-sd(sensi.we3, na.rm = TRUE)
  sd.speci3<-sd(speci.we3, na.rm = TRUE)
  sd.roc3<-sd(roc.we3, na.rm = TRUE)
  sd.accu3<-sd(accu.we3, na.rm = TRUE)
  sd.ppv3<-sd(ppv.we3, na.rm = TRUE)
  sd.npv3<-sd(npv.we3, na.rm = TRUE)
  sd.mcc3<-sd(mcc.we3, na.rm = TRUE)
  
  final.measure3<-c(m.sensi3,sd.sensi3,m.speci3,sd.speci3,m.accu3,sd.accu3,m.ppv3,sd.ppv3,m.npv3,sd.npv3,m.roc3,sd.roc3,m.mcc3,sd.mcc3)
  
  return(list(Maverage=final.measure1, Mmv=final.measure2, Mweight=final.measure3))
}
