library(dplyr)
library(ROSE)
library(caret)
library(mixOmics)
library(mltools)

#PLS Model

plsFunction<-function(y,nt,data,n,n.col,n.comp,k){
  sensi.model<-c()
  speci.model<-c()
  roc.model<-c()
  accu.model<-c()
  ppv.model<-c()
  npv.model<-c()
  mcc.model<-c()
  b = floor(nt*n)
  df.values <- res <- list()
  
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
    spectra.train<-data.matrix(df.train[,4:n.col])
    if(y=="sick"){
      y.train<-df.train$Sick
    }else if(y=="lameness"){
      y.train<-df.train$Lameness
    }else{
      y.train<-df.train$Mastitis
    }
    
    #Test
    df.test<-df[(b+1):n,]
    spectra.test<-data.matrix(df.test[,4:n.col])
    if(y=="sick"){
      y.test<-df.test$Sick
    }else if(y=="lameness"){
      y.test<-df.test$Lameness
    }else{
      y.test<-df.test$Mastitis
    }
    
    #Support Vector Machine Model
    modelo <- mixOmics::plsda(X=spectra.train, Y=y.train, ncomp = 30, scale = FALSE)

    #Predictions
    predicciones <- predict(modelo, newdata=spectra.test, dist = "mahalanobis.dist")$class$mahalanobis.dist[,n.comp]
    
    #Saving the values and probabilities
    df.values[[i]]<-predicciones
    
    #Specificity and Sensitivity
    u <- sort(union(predicciones, y.test), decreasing = FALSE)
    t <- table(factor(y.test, u), factor(predicciones, u))
    sensi.model[i]<-t[2,2]/(t[2,1]+t[2,2])
    speci.model[i]<-t[1,1]/(t[1,1]+t[1,2])
    roc.model[i]<-as.numeric(roc.curve(y.test, predicciones,plotit = F)$auc)
    accu.model[i]<-(t[1,1]+t[2,2])/(t[1,1]+t[1,2]+t[2,1]+t[2,2])
    ppv.model[i]<-t[2,2]/(t[2,2]+t[1,2])
    npv.model[i]<-t[1,1]/(t[1,1]+t[2,1])
    mcc.model[i]<-mcc(as.factor(predicciones),y.test)
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
  return(list(measure=final.measure,df.values=df.values))
}
