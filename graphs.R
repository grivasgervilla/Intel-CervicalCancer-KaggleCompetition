library(ggplot2)

# Representando gráficamente el desbalanceo entre clases

iniciales = c(250,781,450)
extras = c(1394, 3285, 2325)

total = c(extras, iniciales)

df <- data.frame(total, rep(c('Tipo 1', 'Tipo 2', 'Tipo 3'), 2), c(rep("Extras", 3), rep("Base", 3)))
colnames(df) <- c('Cantidad', 'Tipo', 'Extra')

imbalancedPlot <- ggplot(df, aes(x = Tipo, y = Cantidad)) + geom_bar(stat="identity", aes(fill = factor(Extra))) + 
  scale_x_discrete(name = "Tipo") +
  scale_fill_manual(name = "Conjunto", labels = c("Inicial", "Extra"), values =  c("steelblue3", "steelblue4"))  + ylab("cantidad")
imbalancedPlot

# Representando la evolución del val_acc segun las epochs

finetuning = c(0.56276, 0.56983, 0.57042, 0.57572, 0.58014, 0.58073, 0.58427, 0.58603, 0.59075, 0.59104, 0.59428, 0.59723, 0.59959)
finetuningEpochs = c(0,1,6,9,18,25,29,33,36,42,43,48,49)

df <- data.frame(finetuningEpochs, finetuning)
colnames(df) <- c('Epoch', 'Val_acc')

valAccPlot <- ggplot(df, aes(x = Epoch, y = Val_acc)) + geom_line(colour = "steelblue3", size = 1) 
valAccPlot <- valAccPlot + geom_point(colour = "steelblue4", size = 3) +  ylim(0.4,0.8) + ylab("val_acc") + xlab("epoch")
valAccPlot 
