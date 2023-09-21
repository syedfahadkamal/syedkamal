setwd("C:/Users/Syed Fahad/OneDrive/Documents/R ")
library(margins)

idea <- read.csv("idea.csv")

# Answer 1.a)
table(idea$Compulsory.voting)
idea$Compulsory.voting <- factor(idea$Compulsory.voting)

summary(idea$Population_in_million)

mod <- lm(turnout ~ Compulsory.voting + Population_in_million, data = idea)
summary(mod)

# Answer 1.b)
plot(margins(mod),
     labels = c("Compulsory voting", "Population size"),
     ylab = "Effects on electoral turnout",
     main = "Coefficient plot")

cplot(mod, "Compulsory.voting",
      xlab = "Compulsory voting",
      ylab = "Predicted electoral turnout",
      main = "Fitted values plot")

cplot(mod, "Population_in_million",
      xlab = "Population size (in million)",
      ylab = "Predicted electoral turnout",
      main = "Fitted values plot")

idea <- read.csv("idea.csv")

# Answer 2.a)
plot(mod, 2)

shapiro.test(mod$residuals)

# Answer 2.b)
plot(mod, 1)

# Answer 2.c)
plot(mod, 3)

# Answer 3
boxplot(idea$Population_in_million,
        ylab = "Population size (in million)",
        main = "Boxplot of Population Size")

boxplot(mod$residuals,
        ylab = "Model residuals",
        main = "Boxplot of Residuals")
