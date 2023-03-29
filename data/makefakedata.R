library(dplyr)

n <- 10000
newdata_FL <- data.frame(matrix(NA, nrow=n, ncol=0)) %>%
  mutate(vdonor = sample(c('tony', 'wanda', 'natalia', 'steve', 'bruce',
                           'peter', 'thor', 'clint', 'carol', 'pietro'),
                         n,
                         replace=TRUE),
         age = runif(n, min=20, max=60),
         sex = ifelse(vdonor %in% c('wanda','natalia','carol'), 'Women', 'Men'),
         date = sample(seq(as.Date('2012/01/01'), as.Date('2020/01/01'), by='day'), n, replace=TRUE),
         date_of_first_donation = sample(seq(as.Date('2012/01/01'), as.Date('2015/01/01'), by='day'), n, replace=TRUE),
         Hb = runif(n, min=7, max=10),
         Hb_deferral = ifelse((Hb < 7.8 & sex == 'Women') |
                                (Hb < 8.4 & sex == 'Men'), 
                              1, 0),
         height = rnorm(n, 170, 15),
         weight = rnorm(n, 80, 15),
         smoking = sample(c(0,1), n, replace=TRUE, prob=c(0.8,0.2)),
         bmi = weight / height^2,
         snp_17_58358769 = sample(c(0,1,2), n, replace=TRUE),
         snp_6_32617727 = sample(c(0,1,2), n, replace=TRUE),
         snp_15_45095352 = sample(c(0,1,2), n, replace=TRUE),
         snp_1_169549811 = sample(c(0,1,2), n, replace=TRUE),
         prs_anemia = rnorm(n, 0.00001, 0.00015),
         prs_ferritin = rnorm(n, 0.00001, 0.0003),
         prs_hemoglobin = rnorm(n, 0.001, 0.015))

write.csv(newdata_FL, 'fakedata_finland.csv')

newdata_NL <- data.frame(matrix(NA, nrow=n, ncol=0)) %>%
  mutate(KeyID = sample(c('tony', 'wanda', 'natalia', 'steve', 'bruce',
                          'peter', 'thor', 'clint', 'carol', 'pietro'),
                        n,
                        replace=TRUE),
         Sex = ifelse(KeyID %in% c('wanda','natalia','carol'), 'F', 'M'),
         DoB = case_when(KeyID == 'tony' ~ as.Date('1958/01/01'),
                         KeyID == 'wanda' ~ as.Date('1965/12/05'),
                         KeyID == 'natalia' ~ as.Date('1972/06/12'),
                         KeyID == 'steve' ~ as.Date('1983/05/22'),
                         KeyID == 'bruce' ~ as.Date('1970/02/09'),
                         KeyID == 'peter' ~ as.Date('1991/09/21'),
                         KeyID == 'thor' ~ as.Date('1990/08/16'),
                         KeyID == 'clint' ~ as.Date('1967/11/18'),
                         KeyID == 'carol' ~ as.Date('1984/10/28'),
                         KeyID == 'pietro' ~ as.Date('1984/05/31')),
         EIN = 'notimportant',
         Date = sample(seq(as.Date('2012/01/01'), as.Date('2022/01/01'), by='day'), n, replace=TRUE),
         Time = runif(n, min=90, max=200)/10,
         Center = 'alsonotimportant',
         Hb = runif(n, min=7, max=10),
         HbOK = ifelse((Hb < 7.8 & Sex == 'F') |
                       (Hb < 8.4 & Sex == 'M'), 
                        0, 1),
         Volume = ifelse((HbOK  == 1), 
                          500, 0),
         Ferritin = runif(n, min=10, max=100)) %>%
  group_by(KeyID) %>%
  mutate(Ferritin = replace(Ferritin, 
                            sample(row_number(), 
                                   size = 0.8 * n(),
                                   replace=FALSE), 
                            NA)) %>%
  ungroup()

write.csv(newdata_NL, 'fakedata_netherlands.csv')

  
  
  
  
  
  
  
  
  
  


