library(dplyr)
library(ROSE)
library(ranger)
library(caret)
library(mltools)

#Random Forest Function

randomFunction<-function(y,nt,data,n,n.col,rare.class,k,n.tree,m.try){
  sensi.model<-c()
  speci.model<-c()
  roc.model<-c()
  accu.model<-c()
  ppv.model<-c()
  npv.model<-c()
  mcc.model<-c()
  b = floor(nt*n)
  df.values <- res <- list()
  df.proba <- res <- list()
  
  for (i in 1:k) {
    #Data
    df<-data[[i]]
    
    if("MF" %in% colnames(df))
    {
      df<-dplyr::select(df, -MF)
    }
    
    #Data Partition
    ##Train
    df.train<-df[1:b,]
    if(y=="sick"){
      t.train<-dplyr::select(df.train, -Mastitis, -Lameness)
    }else if(y=="lameness"){
      t.train<-dplyr::select(df.train, -Sick, -Mastitis)
    }else{
      t.train<-dplyr::select(df.train, -Sick, -Lameness)
    }
    
    ##Test
    df.test<-df[(b+1):n,]
    if(y=="sick"){
      t.test<-dplyr::select(df.test, -Mastitis, -Lameness)
    }else if(y=="lameness"){
      t.test<-dplyr::select(df.test, -Sick, -Mastitis)
    }else{
      t.test<-dplyr::select(df.test, -Sick, -Lameness)
    }
    
    #Weights
    weights <- rep(0, as.numeric(dim(t.train)[[1]]))
    weights[t.train[,1] == 0] <- rare.class
    weights[t.train[,1] == 1] <- (1-rare.class)
    
    #Random Forest
    if(y=="sick"){
      modelo <- ranger(Sick~., data = t.train, importance="impurity", num.trees = n.tree, mtry = m.try, case.weights=weights, probability = TRUE)
    }else if(y=="lameness"){
      modelo <- ranger(Lameness~., data = t.train, importance="impurity", num.trees = n.tree, mtry = m.try, case.weights=weights, probability = TRUE)
    }else{
      modelo <- ranger(Mastitis~., data = t.train, importance="impurity", num.trees = n.tree, mtry = m.try, case.weights=weights, probability = TRUE)
    }
    
    #Predictions
    predicciones <- predict(modelo, df.test[,4:n.col])[["predictions"]]
    probabilities<-predicciones[,1]
    values<-as.factor(ifelse(predicciones[,1]>=0.5,'0','1'))
    
    #Saving the values and probabilities
    df.values[[i]]<-values
    df.proba[[i]]<-probabilities
    
    #Specificity and Sensitivity
    u <- sort(union(values, t.test[,1]), decreasing = FALSE)
    t <- table(factor(t.test[,1], u), factor(values, u))
    sensi.model[i]<-t[2,2]/(t[2,1]+t[2,2])
    speci.model[i]<-t[1,1]/(t[1,1]+t[1,2])
    roc.model[i]<-as.numeric(roc.curve(t.test[,1], values,plotit = F)$auc)
    accu.model[i]<-(t[1,1]+t[2,2])/(t[1,1]+t[1,2]+t[2,1]+t[2,2])
    ppv.model[i]<-t[2,2]/(t[2,2]+t[1,2])
    npv.model[i]<-t[1,1]/(t[1,1]+t[2,1])
    mcc.model[i]<-mcc(values,t.test[,1])
  }
  
  m.sensi<-mean(sensi.model, na.rm = TRUE)
  m.speci<-mean(speci.model, na.rm = TRUE)
  m.roc<-mean(roc.model, na.rm = TRUE)
  m.accu<-mean(accu.model, na.rm = TRUE)
  m.ppv<-mean(ppv.model, na.rm = TRUE)
  m.npv<-mean(npv.model, na.rm = TRUE)
  m.mcc<-mean(mcc.model, na.rm = TRUE)
  sd.sensi<-sd(sensi.model, na.rm = TRUE)
  sd.speci<-sd(speci.model, na.rm = TRUE)
  sd.roc<-sd(roc.model, na.rm = TRUE)
  sd.accu<-sd(accu.model, na.rm = TRUE)
  sd.ppv<-sd(ppv.model, na.rm = TRUE)
  sd.npv<-sd(npv.model, na.rm = TRUE)
  sd.mcc<-sd(mcc.model, na.rm = TRUE)
  
  final.measure<-c(m.sensi,sd.sensi,m.speci,sd.speci,m.accu,sd.accu,m.ppv,sd.ppv,m.npv,sd.npv,m.roc,sd.roc,m.mcc,sd.mcc)
  return(list(measure=final.measure,df.values=df.values,df.proba=df.proba))
}
