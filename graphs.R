library(ggplot2)

iniciales = c(250,781,450)
extras = c(1394, 3285, 2325)

df <- data.frame(iniciales, extras, c('Tipo 1', 'Tipo 2', 'Tipo 3'))
colnames(df) <- c('Iniciales', 'Extras', 'Tipo')

imbalancedPlot <- ggplot(df, aes(Tipo)) + geom_bar(aes(fill = Iniciales + Extras)) + 
  scale_x_continuous(name = "Tipo", breaks = c(0,1,2), labels = c("0" = "Tipo 1", "1" = "Tipo 2", "2" = "Tipo 3")) +
  scale_fill_manual(name = "Conjunto", labels = c("Inicial", "Extra"), values =  c("khaki3", "rosybrown4"))                   + ylab("#pasajeros")
imbalancedPlot