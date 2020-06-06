#Just run this script once
library(tidyverse)
library(here)
library(ggthemes)
#pull filenames in results folder
filenames <- list.files(here())

#get all of the files for the pre-made tf models/weights. 
#you might need different index
dat <-map(list.files(here())[7:10], read_csv)

#bind dataframes together and get the vars in the correct shape
dat2 <- bind_rows(dat, .id = "column_label") %>% 
  rename(epoch = X1) %>% 
  gather(key = 'acc_name', value = 'acc_value', acc, val_acc) %>% 
  mutate(acc_name = ifelse(acc_name == 'acc', 'Train', 'Validation'))
  #unite('loss', filename, loss_name, sep = ' - - ')

#plot the data for the pre-made architectures
ggplot(data = dat2, aes(x = epoch, y = acc_value, col = model, lty = acc_name))+
  geom_line(size = 1)+
  xlim(0,40)+
  labs(x = 'Training Epoch',
       y = "Accuracy %",
       title = 'Tensorflow Models with Pre-Loaded Weights',
       lty = 'Train and Validation',
       col = 'Model Name')+
  scale_color_ptol()+
  theme_bw()

#Do the same things as above for the simple models
dat <-map(list.files(here(), pattern = '.csv')[1:7], read_csv)

dat2 <- bind_rows(dat, .id = "column_label") %>% 
  rename(epoch = X1) %>% 
  gather(key = 'acc_name', value = 'acc_value', acc, val_acc) %>% 
  mutate(acc_name = ifelse(acc_name == 'acc', 'Train', 'Validation')) %>% 
  mutate(acc_value)

ord <- dat2 %>% 
  filter(acc_name == "Validation") %>% 
  group_by(model) %>% 
  summarize(acc_value = max(acc_value)) %>% 
  arrange(acc_value)

ggplot(data = dat2 %>% 
         mutate(model = ordered(model, 
                                levels = ord$model,
                                labels = c(ord$model[1], 
                                           "16-32-64, Flip-HV, Dropout\n Zoom 0.10, Rotation 15",
                                           ord$model[3:7]))), 
                aes(x = epoch, y = acc_value, col = model, lty = acc_name))+
  geom_line(size = 1)+
  xlim(0,40)+
  labs(x = 'Training Epoch',
       y = "Accuracy %",
       title = 'Simple Networks',
       lty = 'Train and Validation',
       col = 'Models')+
  geom_hline(aes(yintercept = 0.67, col= 'Random Forest Baseline'), size = 1)+
  scale_color_ptol()+
  theme_bw()
