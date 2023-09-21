setwd("C:/Users/Syed Fahad/OneDrive/Documents/R ")
install.packages(c("sp", "raster", "rgdal", "spdep", "tigris", "tidyverse", "dplyr", "sf", "readr", "car", "magrittr"))
install.packages(c("janitor"))
library(sp)
library (spdep)
library (ggplot2)
library (tidyverse)
library (tigris)
library (dplyr)
library(sf)
library(readr)
library(car)
library(magrittr)
library(janitor)

povdf = read_csv("uspoverty2000.csv",
                 col_types = cols(FIPS_N = col_character()),
                 trim_ws = FALSE) %>%
  arrange(as.numeric(as.character(FIPS_N)))
nrow(povdf)
head(povdf)
summary(povdf)
cor(povdf$povty, povdf$feemp)

ggplot(povdf, aes(x=povty)) +
  geom_histogram(aes(y=..density..),fill="grey", col="black") +
  theme_bw() +
  xlab("Poverty Rate") + ylab("Density")

ggplot(povdf, aes(x=feemp, y=povty)) +
  geom_point(size = 0.5) +
  theme_bw() +
  xlab("Female Employment Rate") + ylab("Poverty Rate")

m1 = lm(povty ~ ag + manu + retail + foreign + feemp + hsch + black + hisp,
        data=povdf)
summary(m1)

cbind(coefest = coef(m1), confint(m1))

m2 = step(m1)

n = nrow(povdf)
m3 = step(m1, k=log(n))

summary(m3)

ggplot(m3, aes(sample=rstandard(m3))) +
  geom_qq(size=0.5) +
  stat_qq_line() +
  theme_bw() +
  xlab("Theoretical Quantiles") + ylab("Standardized Residuals")

ggplot(m3,aes(x=.fitted, y=.resid)) +
  geom_point(size=0.5) +
  geom_smooth() +
  theme_bw() +
  xlab("Fitted Values") + ylab("Residuals")

head(povdf)

uscounties <- counties(state = povdf$STATEFP,
                       cb=TRUE, year=2000)
uscounties_sf <- st_as_sf(uscounties)

uscounties_sf$FIPS_N <- paste0(uscounties_sf$STATEFP, uscounties_sf$COUNTYFP)
povdf_uscounties_sf <- sp::merge(uscounties_sf, povdf, by="FIPS_N")

povdf_uscounties_spdf <- as_Spatial(povdf_uscounties_sf,
                                    IDs = povdf_uscounties_sf$FIPS_N)
row.names(povdf_uscounties_spdf) <- row.names(povdf)
pov_nb <- poly2nb(povdf_uscounties_spdf,
                  row.names = povdf_uscounties_spdf$FIPS_N)

Listw_povW = nb2listw(pov_nb, style="W",zero.policy = TRUE)

Listw_povB = nb2listw(pov_nb, style="B", zero.policy = TRUE)

county_geoms <- st_geometry(povdf_uscounties_sf)
cntrd <- st_centroid(county_geoms)
coords <- st_coordinates(cntrd)

moran.test(povdf$povty, Listw_povB, zero.policy = TRUE,
alternative = "two.sided")    

set.seed(1)
moran.mc(povdf$povty, Listw_povB, zero.policy = TRUE, nsim=999)
        
head(round(localmoran(povdf$povty, Listw_povB, zero.policy = TRUE),2))

m3_sar = spautolm(povty ~ ag + foreign + feemp + hsch + black + hisp,
data=povdf, listw = list_povW, zero.policy = TRUE, family="SAR")