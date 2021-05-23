library(dplyr)
library(keras)
library(tensorflow)
library(reticulate)
library(ROSE)
library(caret)
library(mltools)

# Convolutional Neural Network Function

cnnFunction<-function(y,nt,data,n,n.col,k,weights){
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
  
  #Convolutional Neural Network Model
    model <- keras_model_sequential() %>% 
      layer_conv_1d(filters = 84, kernel_size = 7,input_shape = c(858,1)) %>%
      layer_batch_normalization() %>%
      layer_activation_leaky_relu() %>%
      layer_dropout(0.5) %>%
      layer_max_pooling_1d() %>%
      layer_flatten() %>%
      layer_dense(units = 1000) %>%
      layer_batch_normalization() %>%
      layer_activation_leaky_relu() %>%
      layer_dropout(0.5) %>%
      layer_dense(units = 2, activation = "sigmoid")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy")
  )
  
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
    x_train = array_reshape(spectra.train, c(dim(spectra.train), 1))
    if(y=="sick"){
      y.train<-to_categorical(df.train$Sick, 2)
    }else if(y=="lameness"){
      y.train<-to_categorical(df.train$Lameness, 2)
    }else{
      y.train<-to_categorical(df.train$Mastitis, 2)
    }
    
    #Test
    df.test<-df[(b+1):n,]
    spectra.test<-data.matrix(df.test[,4:n.col])
    x_test = array_reshape(spectra.test, c(dim(spectra.test), 1))
    if(y=="sick"){
      y.test<-df.test$Sick
    }else if(y=="lameness"){
      y.test<-df.test$Lameness
    }else{
      y.test<-df.test$Mastitis
    }
    
    #Neural Network Model
    modelo<-model %>% fit(x_train, y.train, batch_size = 40, epochs = 40, validation_split=0.2, class_weight=weights)
    
    #Predictions
    predicciones<-model %>% predict_classes(x_test)
    probab<-model %>% predict(x_test)
    probabilities<-probab[,1]
    
    #Saving the values and probabilities
    df.values[[i]]<-predicciones
    df.proba[[i]]<-probabilities
    
    #Specificity and Sensitivity
    u <- sort(union(predicciones, y.test), decreasing = FALSE)
    t <- table(factor(y.test, u), factor(predicciones, u))
    sensi.model[i]<-t[2,2]/(t[2,1]+t[2,2])
    speci.model[i]<-t[1,1]/(t[1,1]+t[1,2])
    roc.model[i]<-as.numeric(roc.curve(y.test,predicciones,plotit = F)$auc)
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
  return(list(measure=final.measure,df.values=df.values,df.proba=df.proba))
}

cnnFunction.d1<-function(y,nt,data,n,n.col,k,weights){
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
  
  # Convolutional Neural Network Model
  if(y=="sick"){
    model <- keras_model_sequential() %>% 
      layer_conv_1d(filters = 84, kernel_size = 5,input_shape = c(858,1)) %>%
      layer_batch_normalization() %>%
      layer_activation_leaky_relu() %>%
      layer_dropout(0.5) %>%
      layer_max_pooling_1d() %>%
      layer_flatten() %>%
      layer_dense(units = 1000) %>%
      layer_batch_normalization() %>%
      layer_activation_leaky_relu() %>%
      layer_dropout(0.5) %>%
      layer_dense(units = 2, activation = "sigmoid") 
    
  }else{
    model <- keras_model_sequential() %>% 
      layer_conv_1d(filters = 84, kernel_size = 7,input_shape = c(858,1)) %>%
      layer_batch_normalization() %>%
      layer_activation_leaky_relu() %>%
      layer_dropout(0.5) %>%
      layer_max_pooling_1d() %>%
      layer_flatten() %>%
      layer_dense(units = 1000) %>%
      layer_batch_normalization() %>%
      layer_activation_leaky_relu() %>%
      layer_dropout(0.5) %>%
      layer_dense(units = 2, activation = "sigmoid") 
    
  }
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy")
  )
  
  for (i in 1:k) {
    #Data
    df<-data[[i]]
    
    #Data Partition
    ##Train
    df.train<-df[1:b,]
    spectra.train<-as.matrix(df.train[,4:n.col])
    x_train = array_reshape(spectra.train, c(dim(spectra.train), 1))
    if(y=="sick"){
      y.train<-to_categorical(df.train$Sick, 2)
    }else if(y=="lameness"){
      y.train<-to_categorical(df.train$Lameness, 2)
    }else{
      y.train<-to_categorical(df.train$Mastitis, 2)
    }
    
    #Test
    df.test<-df[(b+1):n,]
    spectra.test<-as.matrix(df.test[,4:n.col])
    x_test = array_reshape(spectra.test, c(dim(spectra.test), 1))
    if(y=="sick"){
      y.test<-df.test$Sick
    }else if(y=="lameness"){
      y.test<-df.test$Lameness
    }else{
      y.test<-df.test$Mastitis
    }
    
    #Neural Network Model
    modelo<-model %>% fit(x_train, y.train, batch_size = 40, epochs = 40, validation_split=0.2, class_weight=weights)
    
    #Predictions
    predicciones<-model %>% predict_classes(x_test)
    probab<-model %>% predict(x_test)
    probabilities<-probab[,1]
    
    #Saving the values and probabilities
    df.values[[i]]<-predicciones
    df.proba[[i]]<-probabilities
    
    #Specificity and Sensitivity
    u <- sort(union(predicciones, y.test), decreasing = FALSE)
    t <- table(factor(y.test, u), factor(predicciones, u))
    sensi.model[i]<-t[2,2]/(t[2,1]+t[2,2])
    speci.model[i]<-t[1,1]/(t[1,1]+t[1,2])
    roc.model[i]<-as.numeric(roc.curve(y.test,predicciones,plotit = F)$auc)
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
  return(list(measure=final.measure,df.values=df.values,df.proba=df.proba))
}

cnnFunction.d4<-function(y,nt,data,n,n.col,k,weights){
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
  
  # Convolutional Neural Network Model
  model <- keras_model_sequential() %>% 
    layer_conv_1d(filters = 84, kernel_size = 7,input_shape = c(852,1)) %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.5) %>%
    layer_max_pooling_1d() %>%
    layer_flatten() %>%
    layer_dense(units = 1000) %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.5) %>%
    layer_dense(units = 2, activation = "sigmoid")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy")
  )
  
  for (i in 1:k) {
    #Data
    df<-data[[i]]
    
    #Data Partition
    ##Train
    df.train<-df[1:b,]
    spectra.train<-as.matrix(df.train[,4:n.col])
    x_train = array_reshape(spectra.train, c(dim(spectra.train), 1))
    if(y=="sick"){
      y.train<-to_categorical(df.train$Sick, 2)
    }else if(y=="lameness"){
      y.train<-to_categorical(df.train$Lameness, 2)
    }else{
      y.train<-to_categorical(df.train$Mastitis, 2)
    }
    
    #Test
    df.test<-df[(b+1):n,]
    spectra.test<-as.matrix(df.test[,4:n.col])
    x_test = array_reshape(spectra.test, c(dim(spectra.test), 1))
    if(y=="sick"){
      y.test<-df.test$Sick
    }else if(y=="lameness"){
      y.test<-df.test$Lameness
    }else{
      y.test<-df.test$Mastitis
    }
    
    #Neural Network Model
    modelo<-model %>% fit(x_train, y.train, batch_size = 40, epochs = 40, validation_split=0.2, class_weight=weights)
    
    #Predictions
    predicciones<-model %>% predict_classes(x_test)
    probab<-model %>% predict(x_test)
    probabilities<-probab[,1]
    
    #Saving the values and probabilities
    df.values[[i]]<-predicciones
    df.proba[[i]]<-probabilities
    
    #Specificity and Sensitivity
    u <- sort(union(predicciones, y.test), decreasing = FALSE)
    t <- table(factor(y.test, u), factor(predicciones, u))
    sensi.model[i]<-t[2,2]/(t[2,1]+t[2,2])
    speci.model[i]<-t[1,1]/(t[1,1]+t[1,2])
    roc.model[i]<-as.numeric(roc.curve(y.test,predicciones,plotit = F)$auc)
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
  return(list(measure=final.measure,df.values=df.values,df.proba=df.proba))
}
